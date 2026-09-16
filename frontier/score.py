"""Non-parametric frontier reward and rollout reliability helpers."""

import math
import torch


def quantiles(values):
    values = torch.as_tensor(values).float().flatten()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("frontier calibration values must be finite and nonempty")
    return {
        "q50": torch.quantile(values, 0.50).item(),
        "q95": torch.quantile(values, 0.95).item(),
        "q99": torch.quantile(values, 0.99).item(),
        "q995": torch.quantile(values, 0.995).item(),
    }


def _normalize(value, stats):
    q50 = torch.as_tensor(stats["q50"], device=value.device, dtype=value.dtype)
    q95 = torch.as_tensor(stats["q95"], device=value.device, dtype=value.dtype)
    return ((value - q50) / (q95 - q50 + 1e-6)).clamp(0.0, 2.0)


class FrontierScore:
    """Action-effect frontier score with non-compensatory validity gates."""

    def __init__(self, stats, beta=0.5, invalid_penalty=1.0,
                 use_context_gate=True, use_consistency_gate=True,
                 rollout_reliability=False, prediction_error_limit=None,
                 reliability_decay=0.5, error_fallback_steps=1):
        self.stats = stats
        self.beta = float(beta)
        self.invalid_penalty = float(invalid_penalty)
        self.use_context_gate = bool(use_context_gate)
        self.use_consistency_gate = bool(use_consistency_gate)
        self.rollout_reliability = bool(rollout_reliability)
        self.prediction_error_limit = (
            None if prediction_error_limit is None else float(prediction_error_limit)
        )
        self.reliability_decay = float(reliability_decay)
        self.error_fallback_steps = int(error_fallback_steps)
        if self.beta < 0 or self.invalid_penalty <= 0:
            raise ValueError("beta must be nonnegative and penalty must be positive")
        if self.rollout_reliability and (
            self.prediction_error_limit is None or self.prediction_error_limit <= 0
        ):
            raise ValueError("rollout reliability needs a positive prediction-error limit")
        if not 0 <= self.reliability_decay < 1:
            raise ValueError("reliability decay must be in [0, 1)")
        if self.error_fallback_steps < 0:
            raise ValueError("error fallback steps must be nonnegative")
        self.reset_reliability()

    @property
    def fallback_active(self):
        return self.rollout_reliability and self._fallback_remaining > 0

    def reset_reliability(self):
        self.reliability_inflation = 1.0
        self._fallback_remaining = 0

    def observe_prediction_error(self, error, consumed_fallback=False):
        """Update the next-decision gate using only the realized past error."""
        if not self.rollout_reliability:
            return
        if consumed_fallback and self._fallback_remaining:
            self._fallback_remaining -= 1
        error = float(error)
        if not math.isfinite(error):
            return
        ratio = error / self.prediction_error_limit
        self.reliability_inflation = max(
            1.0,
            self.reliability_decay * self.reliability_inflation
            + (1.0 - self.reliability_decay) * max(1.0, ratio),
        )
        if ratio > 1.0:
            self._fallback_remaining = max(
                self._fallback_remaining, self.error_fallback_steps
            )

    def __call__(self, novelty, ambiguity, effect_consistency, context_distance):
        finite = (torch.isfinite(novelty) & torch.isfinite(ambiguity) &
                  torch.isfinite(effect_consistency) & torch.isfinite(context_distance))
        context_ratio = context_distance / float(self.stats["context_q95"])
        consistency_ratio = effect_consistency / float(self.stats["consistency_q95"])
        context_ok = context_ratio * self.reliability_inflation < 1.0
        consistency_ok = consistency_ratio * self.reliability_inflation < 1.0
        valid = finite.clone()
        if self.use_context_gate:
            valid &= context_ok
        if self.use_consistency_gate:
            valid &= consistency_ok
        novelty_norm = _normalize(novelty, self.stats["novelty"])
        ambiguity_norm = _normalize(ambiguity, self.stats["ambiguity"])
        joint_risk = torch.maximum(context_ratio, consistency_ratio)
        if self.fallback_active:
            valid = finite
            reward = torch.where(
                finite,
                -joint_risk,
                joint_risk.new_full((), -(self.invalid_penalty + 1e3)),
            )
        else:
            reward = novelty_norm + self.beta * ambiguity_norm
            reward = torch.where(valid, reward, reward.new_full((), -self.invalid_penalty))
        return {
            "reward": reward,
            "valid": valid,
            "novelty_norm": novelty_norm,
            "ambiguity_norm": ambiguity_norm,
            "context_ok": context_ok,
            "consistency_ok": consistency_ok,
            "joint_risk": joint_risk,
        }
