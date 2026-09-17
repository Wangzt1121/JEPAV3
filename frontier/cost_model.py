"""CEM-compatible, training-free frontier cost around a frozen LeWM."""

import torch
from torch import nn


class FrontierCostModel(nn.Module):
    """Replace goal cost with a non-parametric action-effect frontier reward."""

    def __init__(self, lewm, memory, scorer, history_size, gamma=0.9,
                 min_valid_ratio=0.6, action_low=None, action_high=None,
                 debug_shapes=False):
        super().__init__()
        self.lewm = lewm.eval().requires_grad_(False)
        self.memory = memory
        self.scorer = scorer
        self.history_size = int(history_size)
        self.gamma = float(gamma)
        self.min_valid_ratio = float(min_valid_ratio)
        self.action_low = action_low
        self.action_high = action_high
        self.debug_shapes = bool(debug_shapes)
        self._printed_shapes = False
        self.last_metrics = None
        self.last_first_prediction = None
        self.action_prior = None
        self.action_prior_sigma = None
        self.action_prior_max_std = None
        self.action_prior_penalty = 0.0
        if not 0 <= self.gamma <= 1 or not 0 <= self.min_valid_ratio <= 1:
            raise ValueError("gamma and min_valid_ratio must be in [0, 1]")
        if (action_low is None) != (action_high is None):
            raise ValueError("provide both action bounds or neither")

    def set_action_prior(self, action, sigma, max_deviation_std, penalty):
        """Constrain future CEM actions to a Gaussian expert-action neighborhood."""
        action = torch.as_tensor(action).float()
        sigma = torch.as_tensor(sigma).float()
        if action.ndim != 3 or sigma.ndim not in (1, 3):
            raise ValueError("action prior must be B,T,A and sigma must be A or B,T,A")
        if sigma.ndim == 1:
            sigma = sigma.reshape(1, 1, -1).expand_as(action)
        if sigma.shape != action.shape or action.shape[-1] == 0:
            raise ValueError("action prior and sigma shapes differ")
        if not torch.isfinite(action).all() or not torch.isfinite(sigma).all():
            raise ValueError("action prior and sigma must be finite")
        if (sigma <= 0).any() or max_deviation_std <= 0 or penalty < 0:
            raise ValueError("expert noise parameters must be positive")
        self.action_prior = action
        self.action_prior_sigma = sigma
        self.action_prior_max_std = float(max_deviation_std)
        self.action_prior_penalty = float(penalty)

    def _prior_deviation(self, info_dict, action_candidates):
        if self.action_prior is None:
            return None, None
        prefix = info_dict["pixels"].shape[2] - 1
        future = action_candidates[:, :, prefix:]
        prior = self.action_prior.to(future).unsqueeze(1)
        sigma = self.action_prior_sigma.to(future).unsqueeze(1)
        if prior.shape[0] != future.shape[0] or prior.shape[2:] != future.shape[2:]:
            raise ValueError("expert action prior does not match CEM candidates")
        deviation = (future - prior) / sigma
        penalty = deviation.pow(2).mean(dim=(-1, -2))
        outside = deviation.abs().amax(dim=(-1, -2)) > self.action_prior_max_std
        return penalty, outside

    def prepare_candidates(self, info_dict, action_candidates):
        """Insert observed past actions into the candidate sequence."""
        if action_candidates.ndim != 4 or info_dict["pixels"].ndim != 6:
            raise ValueError("expected actions B,S,T,A and pixels B,S,H,C,Y,X")
        batch, samples, steps, action_dim = action_candidates.shape
        if info_dict["pixels"].shape[:2] != (batch, samples):
            raise ValueError("candidate and observation dimensions differ")
        initial_history = info_dict["pixels"].shape[2]
        if initial_history < self.history_size or steps < initial_history:
            raise ValueError("candidate sequence is too short for the history")
        actions = action_candidates.clone()
        if self.action_low is not None:
            low = torch.as_tensor(self.action_low, device=actions.device, dtype=actions.dtype)
            high = torch.as_tensor(self.action_high, device=actions.device, dtype=actions.dtype)
            actions = actions.clamp(min=low, max=high)
        prefix = initial_history - 1
        if self.action_prior is not None:
            prior = self.action_prior.to(actions).unsqueeze(1)
            sigma = self.action_prior_sigma.to(actions).unsqueeze(1)
            future = actions[:, :, prefix:]
            if prior.shape[0] != batch or prior.shape[2:] != future.shape[2:]:
                raise ValueError("expert action prior does not match planning horizon")
            radius = self.action_prior_max_std * sigma
            actions[:, :, prefix:] = torch.maximum(
                torch.minimum(future, prior + radius), prior - radius
            )
        if prefix:
            past = info_dict.get("action")
            if past is None or past.shape[:2] != (batch, samples):
                raise ValueError("observed past action blocks are required")
            if past.shape[2] < prefix or past.shape[-1] != action_dim:
                raise ValueError("observed past actions are not aligned")
            actions[:, :, :prefix] = past[:, :, :prefix].to(actions)
        if not torch.isfinite(actions).all():
            raise ValueError("candidate actions must be finite")
        return actions

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        device = next(self.lewm.parameters()).device
        info = {key: value.clone().to(device) if torch.is_tensor(value) else value
                for key, value in info_dict.items()}
        action_candidates = action_candidates.to(device)
        prior_penalty, outside_prior = self._prior_deviation(info, action_candidates)
        actions = self.prepare_candidates(info, action_candidates)
        batch, samples, steps, action_dim = actions.shape
        initial_history = info["pixels"].shape[2]
        rollout = self.lewm.rollout(info, actions, history_size=self.history_size)
        states = rollout["predicted_emb"]
        if states.shape[:3] != (batch, samples, steps + 1):
            raise ValueError(f"unexpected LeWM rollout shape: {tuple(states.shape)}")
        action_emb = self.lewm.action_encoder(
            actions.reshape(batch * samples, steps, action_dim)
        ).reshape(batch, samples, steps, -1)

        first = initial_history - 1
        z_t = states[:, :, first:-1]
        z_next = states[:, :, first + 1:]
        a_t = action_emb[:, :, first:]
        steps_scored = z_t.shape[2]
        stats = self.memory.query(z_t, a_t, z_next - z_t)
        score = self.scorer(
            stats["novelty"], stats["local_ambiguity"],
            stats["effect_consistency"], stats["context_distance"],
        )
        discount = self.gamma ** torch.arange(
            steps_scored, device=device, dtype=z_t.dtype
        )
        reward = (score["reward"] * discount).sum(dim=-1)
        valid_ratio = score["valid"].float().mean(dim=-1)
        reward = torch.where(
            valid_ratio >= self.min_valid_ratio,
            reward,
            reward.new_full((), -self.scorer.invalid_penalty),
        )
        cost = -reward
        if prior_penalty is not None:
            cost = cost + self.action_prior_penalty * prior_penalty
            cost = torch.where(
                outside_prior,
                cost.new_full((), self.scorer.invalid_penalty + 1e3),
                cost,
            )
        if cost.shape != (batch, samples) or not torch.isfinite(cost).all():
            raise ValueError(f"invalid CEM cost: shape={tuple(cost.shape)}")

        self.last_first_prediction = z_next[:, :, 0].detach()
        self.last_metrics = {
            "novelty_raw": stats["novelty"].mean(-1).detach(),
            "novelty_norm": score["novelty_norm"].mean(-1).detach(),
            "ambiguity_raw": stats["local_ambiguity"].mean(-1).detach(),
            "ambiguity_norm": score["ambiguity_norm"].mean(-1).detach(),
            "effect_consistency": stats["effect_consistency"].mean(-1).detach(),
            "context_distance": stats["context_distance"].mean(-1).detach(),
            "valid_ratio": valid_ratio.detach(),
            "frontier_reward": reward.detach(),
        }
        if prior_penalty is not None:
            self.last_metrics["expert_deviation_std"] = prior_penalty.sqrt().detach()
            self.last_metrics["expert_prior_valid"] = (~outside_prior).float().detach()
        if self.debug_shapes and not self._printed_shapes:
            for name, value in (
                ("initial_history", info["pixels"]),
                ("action_candidates", actions),
                ("predicted_emb", states),
                ("z_t", z_t),
                ("z_next", z_next),
                ("action_emb", a_t),
                ("context", self.memory._normalize(torch.cat([
                    z_t, a_t
                ], dim=-1))),
                ("transition_descriptor", self.memory._normalize(torch.cat([
                    z_t, a_t, z_next - z_t
                ], dim=-1))),
                ("knn_similarity", stats["novelty"]),
                ("novelty", stats["novelty"]),
                ("ambiguity", stats["local_ambiguity"]),
                ("consistency", stats["effect_consistency"]),
                ("context_distance", stats["context_distance"]),
                ("frontier_reward", score["reward"]),
                ("cost", cost),
            ):
                print(f"{name:24s} = {tuple(value.shape)}")
            print(f"valid_ratio={valid_ratio.mean().item():.6f} finite_cost=True")
            self._printed_shapes = True
        return cost

    def forward(self, info_dict, action_candidates):
        return self.get_cost(info_dict, action_candidates)
