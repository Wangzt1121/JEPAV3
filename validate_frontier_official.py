#!/usr/bin/env python3
"""Paired validation of the exact Frontier sampler used for data collection."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import numpy as np
import pandas as pd
import stable_worldmodel as swm
import torch
from stable_worldmodel.solver.cem import CEMSolver

from frontier.cost_model import FrontierCostModel
from frontier.memory import ActionEffectMemory
from frontier.score import FrontierScore
from utils import get_img_preprocessor


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-states", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--max-noise-std", type=float, default=2.5)
    parser.add_argument("--prior-penalty", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--beta-ambiguity", type=float, default=0.5)
    parser.add_argument("--invalid-penalty", type=float, default=1.0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.6)
    parser.add_argument(
        "--prediction-error-limit", type=float, default=0.021342629566788673
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.test_states < 1 or args.bootstrap_samples < 1:
        parser.error("test-states and bootstrap-samples must be positive")
    if not 0 < args.topk <= args.num_samples:
        parser.error("topk must be in [1, num-samples]")
    if args.noise_sigma <= 0 or args.max_noise_std <= 0 or args.prior_penalty < 0:
        parser.error("expert noise parameters are invalid")
    return args


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def select_test_refs(episode_ids, lengths, count, history, frameskip, rng):
    """Choose at most one aligned transition from each held-out test episode."""
    episodes = np.asarray(episode_ids, dtype=np.int64).copy()
    rng.shuffle(episodes)
    refs = []
    min_t = (history - 1) * frameskip
    for episode in episodes:
        starts = np.arange(min_t, int(lengths[episode]) - frameskip, frameskip)
        if starts.size:
            refs.append((int(episode), int(rng.choice(starts))))
        if len(refs) == count:
            break
    return refs


def bootstrap_paired_difference(frontier, random_values, rng, samples):
    frontier = np.asarray(frontier, dtype=np.float64)
    random_values = np.asarray(random_values, dtype=np.float64)
    difference = frontier - random_values
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "median": float(np.median(difference)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_probability_gt_zero": float((means > 0).mean()),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bootstrap_rng = np.random.default_rng(args.seed + 1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    bank = ActionEffectMemory.load(args.bank, device="cpu")
    stats = torch.load(args.stats, map_location="cpu", weights_only=True)
    metadata = bank.metadata
    if stats.get("format_version") != 2 or metadata.get("sampling_version") != 2:
        raise ValueError(
            "Bank/stats use the obsolete sampler; rebuild with build_frontier_bank.py"
        )
    if not metadata.get("test_episodes"):
        raise ValueError("Bank metadata has no strictly held-out test episodes")
    if metadata.get("action_prior_version") != 1:
        raise ValueError("Bank has no retrievable expert action prior")
    if Path(metadata["dataset"]).resolve() != args.dataset.resolve():
        raise ValueError("validation dataset differs from the Bank source dataset")
    if str(metadata["checkpoint"]) != str(args.checkpoint_dir):
        raise ValueError("validation checkpoint differs from the Bank checkpoint")

    history = int(metadata["history_size"])
    frameskip = int(metadata["frameskip"])
    image_size = int(metadata["img_size"])
    log(f"Loading production LeWM on {device}")
    lewm = swm.wm.utils.load_pretrained(str(args.checkpoint_dir)).to(device).eval()
    lewm.requires_grad_(False)
    scorer = FrontierScore(
        stats,
        beta=args.beta_ambiguity,
        invalid_penalty=args.invalid_penalty,
        use_context_gate=True,
        use_consistency_gate=True,
        rollout_reliability=True,
        prediction_error_limit=args.prediction_error_limit,
        reliability_decay=0.5,
        error_fallback_steps=1,
    )

    action_stats = metadata["action_stats"]
    mean = torch.as_tensor(action_stats["mean"]).cpu().numpy()
    std = torch.as_tensor(action_stats["std"]).cpu().numpy()
    world = swm.World(
        "swm/PushT-v1",
        num_envs=1,
        image_shape=(image_size, image_size),
        max_episode_steps=1000,
        goal_conditioned=True,
    )
    action_space = world.envs.single_action_space
    low = np.asarray(action_space.low)
    high = np.asarray(action_space.high)
    if mean.shape != low.shape:
        world.close()
        raise ValueError("environment action dimension differs from the Bank")
    normalized_low = np.tile((low - mean) / std, frameskip)
    normalized_high = np.tile((high - mean) / std, frameskip)
    raw_sigma = np.full_like(mean, args.noise_sigma, dtype=np.float32)
    normalized_sigma = np.tile(raw_sigma / std, frameskip).astype(np.float32)
    cost_model = FrontierCostModel(
        lewm,
        bank,
        scorer,
        history,
        gamma=args.gamma,
        min_valid_ratio=args.min_valid_ratio,
        action_low=normalized_low,
        action_high=normalized_high,
    ).to(device).eval()
    plan_config = swm.PlanConfig(
        horizon=history,
        receding_horizon=1,
        history_len=history,
        action_block=frameskip,
        warm_start=False,
    )
    solver = CEMSolver(
        cost_model,
        batch_size=1,
        num_samples=args.num_samples,
        topk=args.topk,
        n_steps=args.iterations,
        var_scale=float(np.sqrt(np.mean(normalized_sigma ** 2))),
        device=device,
        seed=args.seed,
    )
    solver.configure(
        action_space=world.envs.action_space,
        n_envs=1,
        config=plan_config,
    )
    image_transform = get_img_preprocessor("pixels", "pixels", image_size)

    def pixels(frames):
        value = torch.from_numpy(np.stack(frames))
        if value.ndim != 4:
            raise ValueError(f"expected an image batch, got {tuple(value.shape)}")
        if value.shape[-1] in (1, 3):
            value = value.permute(0, 3, 1, 2)
        transformed = image_transform({"pixels": value})["pixels"]
        return transformed.unsqueeze(0).to(device)

    def planning_info(history_pixels, past_blocks):
        return {
            "pixels": pixels(history_pixels),
            "action": torch.as_tensor(
                np.stack(past_blocks), device=device, dtype=torch.float32
            ).unsqueeze(0),
        }

    def evaluate_sequence(info, normalized_block):
        expanded = {key: value.unsqueeze(1) for key, value in info.items()}
        sequence = torch.zeros(
            (1, 1, history, normalized_block.size),
            device=device,
            dtype=torch.float32,
        )
        sequence[:, :, -1] = torch.as_tensor(normalized_block, device=device)
        sequence = cost_model.prepare_candidates(expanded, sequence)
        cost_model.get_cost(expanded, sequence)
        return (
            {key: float(value[0, 0]) for key, value in cost_model.last_metrics.items()},
            cost_model.last_first_prediction[0, 0].clone(),
        )

    def execute_and_measure(env, state, goal_state, seed, raw_block, normalized_block):
        env.reset(
            seed=seed,
            options={"state": state.copy(), "goal_state": goal_state.copy()},
        )
        start_pixels = np.asarray(env.render()).copy()
        terminated = False
        for action in raw_block:
            _, _, done, truncated, _ = env.step(action.astype(np.float32))
            terminated = terminated or bool(done or truncated)
            if terminated:
                break
        next_pixels = np.asarray(env.render()).copy()
        real = lewm.encode({"pixels": pixels([start_pixels, next_pixels])})["emb"]
        action_embedding = lewm.action_encoder(
            torch.as_tensor(normalized_block, device=device).reshape(1, 1, -1)
        )[:, 0]
        metrics = bank.query(
            real[:, 0], action_embedding, real[:, 1] - real[:, 0]
        )
        return (
            {key: float(value[0]) for key, value in metrics.items()},
            real[:, 1],
            terminated,
        )

    records = []
    started = time.time()
    envs = {
        "frontier": gym.make(
            "swm/PushT-v1", resolution=image_size, render_mode="rgb_array"
        ),
        "expert_noise": gym.make(
            "swm/PushT-v1", resolution=image_size, render_mode="rgb_array"
        ),
    }
    try:
        with h5py.File(args.dataset, "r", swmr=True) as data:
            required = {"ep_len", "ep_offset", "pixels", "action", "state"}
            missing = sorted(required - set(data))
            if missing:
                raise KeyError(f"dataset is missing columns: {missing}")
            lengths = np.asarray(data["ep_len"][:], dtype=np.int64)
            offsets = np.asarray(data["ep_offset"][:], dtype=np.int64)
            refs = select_test_refs(
                metadata["test_episodes"],
                lengths,
                args.test_states,
                history,
                frameskip,
                rng,
            )
            if not refs:
                raise ValueError("no held-out test transition is available")
            if len(refs) < args.test_states:
                log(f"Requested {args.test_states} states; using {len(refs)}")

            goal_rows = offsets + lengths - 1
            goal_states = np.asarray(data["state"][goal_rows], dtype=np.float64)
            for state_id, (episode, step) in enumerate(refs):
                row = int(offsets[episode]) + step
                history_rows = [
                    row - (history - 1 - index) * frameskip
                    for index in range(history)
                ]
                history_pixels = np.asarray(data["pixels"][history_rows])
                past_blocks = []
                for start in history_rows[:-1]:
                    raw = np.asarray(
                        data["action"][start:start + frameskip], dtype=np.float32
                    )
                    if not np.isfinite(raw).all():
                        raise ValueError(
                            f"nonfinite past action at episode={episode}, step={step}"
                        )
                    past_blocks.append(((raw - mean) / std).reshape(-1))
                info = planning_info(history_pixels, past_blocks)

                current_z = lewm.encode({"pixels": info["pixels"]})["emb"][:, -1]
                expert_normalized_tensor, expert_similarity = bank.nearest_action(
                    current_z
                )
                expert_normalized = expert_normalized_tensor[0].cpu().numpy()
                expert_raw = np.clip(
                    expert_normalized.reshape(frameskip, -1) * std + mean,
                    low,
                    high,
                ).astype(np.float32)
                cost_model.set_action_prior(
                    expert_normalized_tensor.unsqueeze(1),
                    torch.as_tensor(normalized_sigma, device=device),
                    args.max_noise_std,
                    args.prior_penalty,
                )
                init_action = torch.cat([
                    info["action"], expert_normalized_tensor.unsqueeze(1)
                ], dim=1)

                scorer.reset_reliability()
                result = solver(info, init_action=init_action)
                expanded = {key: value.unsqueeze(1) for key, value in info.items()}
                frontier_sequence = cost_model.prepare_candidates(
                    expanded, result["actions"].to(device).unsqueeze(1)
                )
                cost_model.get_cost(expanded, frontier_sequence)
                frontier_predicted = {
                    key: float(value[0, 0])
                    for key, value in cost_model.last_metrics.items()
                }
                frontier_next = cost_model.last_first_prediction[0, 0].clone()
                frontier_normalized = (
                    frontier_sequence[0, 0, history - 1].cpu().numpy()
                )
                frontier_raw = np.clip(
                    frontier_normalized.reshape(frameskip, -1) * std + mean,
                    low,
                    high,
                ).astype(np.float32)

                expert_noise = rng.normal(0.0, raw_sigma, size=expert_raw.shape)
                expert_noise = np.clip(
                    expert_noise,
                    -args.max_noise_std * raw_sigma,
                    args.max_noise_std * raw_sigma,
                )
                expert_noise_raw = np.clip(
                    expert_raw + expert_noise, low, high
                ).astype(np.float32)
                expert_noise_normalized = (
                    (expert_noise_raw - mean) / std
                ).reshape(-1)
                scorer.reset_reliability()
                expert_noise_predicted, expert_noise_next = evaluate_sequence(
                    info, expert_noise_normalized
                )

                state = np.asarray(data["state"][row], dtype=np.float64)
                goal_state = goal_states[episode]
                candidates = (
                    (
                        "frontier",
                        frontier_raw,
                        frontier_normalized,
                        frontier_predicted,
                        frontier_next,
                    ),
                    (
                        "expert_noise",
                        expert_noise_raw,
                        expert_noise_normalized,
                        expert_noise_predicted,
                        expert_noise_next,
                    ),
                )
                for method, raw_block, normalized_block, predicted, predicted_next in candidates:
                    actual, actual_next, terminated = execute_and_measure(
                        envs[method],
                        state,
                        goal_state,
                        args.seed + episode,
                        raw_block,
                        normalized_block,
                    )
                    record = {
                        "state_id": state_id,
                        "episode": episode,
                        "step": step,
                        "method": method,
                        "expert_similarity": expert_similarity[0].item(),
                        "action_noise_rms": float(
                            np.sqrt(np.mean((raw_block - expert_raw) ** 2))
                        ),
                        "real_novelty": actual["novelty"],
                        "real_ambiguity": actual["local_ambiguity"],
                        "real_effect_consistency": actual["effect_consistency"],
                        "real_context_distance": actual["context_distance"],
                        "prediction_error": float(
                            (predicted_next - actual_next[0]).pow(2).mean()
                        ),
                        "terminated_within_block": terminated,
                        **{
                            f"predicted_{key}": value
                            for key, value in predicted.items()
                        },
                    }
                    records.append(record)
                if (state_id + 1) % 10 == 0 or state_id == 0:
                    log(f"Formal CEM validation {state_id + 1}/{len(refs)}")
    finally:
        for env in envs.values():
            env.close()
        world.close()

    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "paired_results.csv", index=False)
    pivots = {
        metric: frame.pivot(index="state_id", columns="method", values=metric)
        for metric in ("real_novelty", "real_ambiguity", "prediction_error")
    }
    summary = {
        "samples": int(frame.state_id.nunique()),
        "method_means": frame.groupby("method")[
            ["real_novelty", "real_ambiguity", "prediction_error"]
        ].mean().to_dict(orient="index"),
        "paired_frontier_minus_expert_noise": {
            metric: bootstrap_paired_difference(
                pivot["frontier"].to_numpy(),
                pivot["expert_noise"].to_numpy(),
                bootstrap_rng,
                args.bootstrap_samples,
            )
            for metric, pivot in pivots.items()
        },
    }
    manifest = {
        "sampler": "nearest expert action + Gaussian noise + FrontierCostModel + CEMSolver",
        "distance": "ActionEffectMemory cosine KNN",
        "score": "normalized novelty - beta * normalized ambiguity",
        "bank": str(args.bank.resolve()),
        "stats": str(args.stats.resolve()),
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint_dir.resolve()),
        "test_episode_source": "strictly held-out test_episodes in Bank metadata",
        "cem": {
            "num_samples": args.num_samples,
            "topk": args.topk,
            "iterations": args.iterations,
            "var_scale": float(np.sqrt(np.mean(normalized_sigma ** 2))),
            "horizon_model_steps": 1,
        },
        "expert_noise": {
            "sigma_raw_action": args.noise_sigma,
            "max_deviation_std": args.max_noise_std,
            "prior_penalty": args.prior_penalty,
            "baseline": "nearest expert action plus one Gaussian perturbation",
        },
        "runtime_seconds": time.time() - started,
        "seed": args.seed,
    }
    (args.output / "summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    log(json.dumps(summary["paired_frontier_minus_expert_noise"], indent=2))


if __name__ == "__main__":
    main()
