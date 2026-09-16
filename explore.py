"""Training-free Action-Effect Frontier exploration with the original CEM."""

from collections import deque
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from stable_worldmodel.solver.cem import CEMSolver

from frontier.cost_model import FrontierCostModel
from frontier.memory import ActionEffectMemory
from frontier.score import FrontierScore
from utils import get_img_preprocessor


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _observation(info):
    pixels = _numpy(info["pixels"])[0, 0].copy()
    if pixels.ndim != 3:
        raise ValueError("environment pixels must be an image")
    if pixels.shape[-1] != 3 and pixels.shape[0] == 3:
        pixels = pixels.transpose(1, 2, 0).copy()
    if pixels.shape[-1] != 3:
        raise ValueError("expected RGB environment pixels")
    result = {"pixels": pixels}
    for key in ("state", "proprio"):
        if key in info:
            result[key] = _numpy(info[key])[0, 0].copy()
    return result


@hydra.main(version_base=None, config_path="./config/explore", config_name="pusht")
@torch.no_grad()
def run(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))
    method = str(cfg.method)
    if method not in ("aefe", "random"):
        raise ValueError("method must be 'aefe' or 'random'")
    device = torch.device(str(cfg.device))
    bank = ActionEffectMemory.load(cfg.artifacts.bank, device="cpu")
    stats = torch.load(cfg.artifacts.stats, map_location="cpu", weights_only=True)
    if stats.get("format_version") != 1:
        raise ValueError("frontier artifacts were not built by build_frontier_bank.py")
    metadata = bank.metadata
    history = int(metadata["history_size"] if cfg.history_size is None else cfg.history_size)
    for key, value in (("checkpoint", str(cfg.checkpoint)),
                       ("history_size", history), ("frameskip", int(cfg.frameskip)),
                       ("img_size", int(cfg.img_size))):
        if metadata.get(key) != value:
            raise ValueError(f"configuration differs from frontier bank: {key}")
    if metadata.get("descriptor_type") != "action_effect":
        raise ValueError("the bank must contain action-effect transitions")
    lewm = swm.wm.utils.load_pretrained(cfg.checkpoint).to(device).eval()
    lewm.requires_grad_(False)
    scorer = FrontierScore(
        stats, beta=float(cfg.frontier.beta_ambiguity),
        invalid_penalty=float(cfg.frontier.invalid_penalty),
        use_context_gate=bool(cfg.frontier.use_context_gate),
        use_consistency_gate=bool(cfg.frontier.use_consistency_gate),
        rollout_reliability=bool(cfg.frontier.rollout_reliability),
        prediction_error_limit=float(cfg.frontier.prediction_error_limit),
        reliability_decay=float(cfg.frontier.reliability_decay),
        error_fallback_steps=int(cfg.frontier.error_fallback_steps),
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world = swm.World(
        str(cfg.world.env_name), num_envs=1,
        image_shape=(int(cfg.img_size), int(cfg.img_size)),
        max_episode_steps=int(cfg.world.max_episode_steps), goal_conditioned=True,
    )
    image_transform = get_img_preprocessor("pixels", "pixels", int(cfg.img_size))
    action_stats = metadata["action_stats"]
    mean = action_stats["mean"].numpy()
    std = action_stats["std"].numpy()
    action_space = world.envs.single_action_space
    low, high = np.asarray(action_space.low), np.asarray(action_space.high)
    block = int(cfg.frameskip)
    if mean.shape != low.shape:
        world.close()
        raise ValueError("environment action dimension differs from the bank")
    normalized_low = np.tile((low - mean) / std, block)
    normalized_high = np.tile((high - mean) / std, block)
    model = FrontierCostModel(
        lewm, bank, scorer, history,
        gamma=float(cfg.planning.gamma),
        min_valid_ratio=float(cfg.frontier.min_valid_ratio),
        action_low=normalized_low, action_high=normalized_high,
        debug_shapes=bool(cfg.debug_frontier),
    ).to(device).eval()
    plan_config = swm.PlanConfig(
        horizon=history - 1 + int(cfg.planning.horizon),
        receding_horizon=1, history_len=history, action_block=block, warm_start=False,
    )
    solver = CEMSolver(
        model, batch_size=1, num_samples=int(cfg.planning.num_samples),
        topk=int(cfg.planning.topk), n_steps=int(cfg.planning.iterations),
        var_scale=float(cfg.planning.var_scale), device=device, seed=int(cfg.seed),
    )
    solver.configure(action_space=world.envs.action_space, n_envs=1, config=plan_config)

    def pixels(frames):
        value = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        return image_transform({"pixels": value})["pixels"].unsqueeze(0).to(device)

    frames = deque(maxlen=history)
    past_actions = deque(maxlen=max(1, history - 1))
    rows, steps, episode = [], 0, 0
    try:
        with HDF5Writer(Path(cfg.output_dataset), mode="error") as writer:
            _, info = world.envs.reset(seed=int(cfg.seed))
            current = _observation(info)
            frames.append(current["pixels"])
            while steps < int(cfg.exploration.total_steps):
                selected_metrics = None
                predicted_next = None
                used_error_fallback = False
                if len(frames) < history or method == "random":
                    selected = rng.uniform(low, high, size=(block, *low.shape)).astype(np.float32)
                else:
                    used_error_fallback = scorer.fallback_active
                    planning_info = {"pixels": pixels(list(frames))}
                    past = (torch.as_tensor(np.stack(past_actions), device=device)
                            if past_actions else torch.empty(0, normalized_low.size, device=device))
                    planning_info["action"] = past.unsqueeze(0)
                    expanded = {key: value.unsqueeze(1) for key, value in planning_info.items()}
                    result = solver(planning_info)
                    sequence = model.prepare_candidates(
                        expanded, result["actions"].to(device).unsqueeze(1)
                    )
                    model.get_cost(expanded, sequence)
                    selected_metrics = model.last_metrics
                    predicted_next = model.last_first_prediction[0, 0]
                    selected = sequence[0, 0, history - 1].cpu().numpy().reshape(block, -1)
                    selected = np.clip(selected * std + mean, low, high).astype(np.float32)

                start_pixels = current["pixels"].copy()
                executed, terminated = [], False
                for action in selected:
                    if steps >= int(cfg.exploration.total_steps):
                        break
                    rows.append({key: value.copy() for key, value in current.items()} | {"action": action.copy()})
                    _, _, dead, truncated, info = world.envs.step(action[None])
                    current = _observation(info)
                    executed.append(action.copy())
                    steps += 1
                    terminated = bool(dead[0] or truncated[0])
                    if terminated:
                        break

                real_prediction_error = float("nan")
                if len(executed) == block:
                    normalized = ((np.stack(executed) - mean) / std).reshape(1, 1, -1).astype(np.float32)
                    real = lewm.encode({"pixels": pixels([start_pixels, current["pixels"]])})["emb"]
                    real_action = lewm.action_encoder(torch.as_tensor(normalized, device=device))[:, 0]
                    if bool(cfg.memory_bank.online_update):
                        bank.add(real[:, 0], real_action, real[:, 1] - real[:, 0])
                    if predicted_next is not None:
                        real_prediction_error = (predicted_next - real[:, 1]).pow(2).mean().item()
                        scorer.observe_prediction_error(
                            real_prediction_error,
                            consumed_fallback=used_error_fallback,
                        )
                    frames.append(current["pixels"])
                    past_actions.append(normalized[0, 0])
                if selected_metrics is not None:
                    print({key: float(value.mean()) for key, value in selected_metrics.items()})
                    print(f"real_prediction_error={real_prediction_error:.8f} bank_size={len(bank)} steps={steps}")
                    print(
                        f"reliability_inflation={scorer.reliability_inflation:.6f} "
                        f"error_fallback_next={scorer.fallback_active}"
                    )
                if terminated:
                    if rows:
                        terminal = {**current, "action": np.full_like(mean, np.nan, dtype=np.float32)}
                        episode_rows = rows + [terminal]
                        writer.write_episode({key: [row[key] for row in episode_rows] for key in terminal})
                        rows.clear()
                    episode += 1
                    if steps < int(cfg.exploration.total_steps):
                        _, info = world.envs.reset(seed=int(cfg.seed) + episode)
                        current = _observation(info)
                        frames.clear(); past_actions.clear(); frames.append(current["pixels"])
                        scorer.reset_reliability()
            if rows:
                terminal = {**current, "action": np.full_like(mean, np.nan, dtype=np.float32)}
                episode_rows = rows + [terminal]
                writer.write_episode({key: [row[key] for row in episode_rows] for key in terminal})
    finally:
        world.close()
        bank.save(cfg.output_bank)
    print(f"real_steps={steps} bank_size={len(bank)} dataset={cfg.output_dataset}")


if __name__ == "__main__":
    run()
