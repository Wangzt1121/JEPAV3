#!/usr/bin/env python3
"""One-shot, training-free Frontier validation for the official PushT LeWM.

The script is intentionally standalone. It does not import or modify the
repository's experimental Frontier implementation. It builds a deduplicated
transition bank, calibrates on held-out episodes, and performs a paired
counterfactual action-ranking test in the PushT simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from scipy.stats import spearmanr

import stable_worldmodel  # noqa: F401 - registers swm/PushT-v1


FRAME_SKIP = 5
HISTORY = 3


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_key_to_current(key: str) -> str:
    """Rename legacy Transformers ViT keys without changing tensor values."""
    if not key.startswith("encoder.encoder.layer."):
        return key
    key = key.replace("encoder.encoder.layer.", "encoder.layers.")
    replacements = {
        ".attention.attention.query.": ".attention.q_proj.",
        ".attention.attention.key.": ".attention.k_proj.",
        ".attention.attention.value.": ".attention.v_proj.",
        ".attention.output.dense.": ".attention.o_proj.",
        ".intermediate.dense.": ".mlp.fc1.",
        ".output.dense.": ".mlp.fc2.",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return key


def load_official_model(checkpoint_dir: Path, device: torch.device):
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "weights.pt"
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    model = instantiate(config)
    original = torch.load(weights_path, map_location="cpu", weights_only=True)
    converted = {official_key_to_current(k): v for k, v in original.items()}
    if len(converted) != len(original):
        raise RuntimeError("Checkpoint key conversion produced a collision")
    model.load_state_dict(converted, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, {
        "config_sha256": sha256(config_path),
        "weights_sha256": sha256(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "state_dict_tensors": len(original),
        "compatibility_key_renames": sum(k != official_key_to_current(k) for k in original),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def preprocess_images(images: np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.asarray(images)
    tensor = torch.from_numpy(array)
    if tensor.ndim != 4:
        raise ValueError(f"Expected four-dimensional image batch, got {tuple(tensor.shape)}")
    if tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    elif tensor.shape[1] not in (1, 3):
        raise ValueError(f"Cannot infer image channel axis from {tuple(tensor.shape)}")
    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if array.dtype == np.uint8 or float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    if tensor.shape[-2:] != (224, 224):
        tensor = F.interpolate(
            tensor, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
        )
    return tensor


@torch.inference_mode()
def encode_image_array(model, images: np.ndarray, device: torch.device) -> torch.Tensor:
    pixels = preprocess_images(images, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model.encoder(pixels, interpolate_pos_encoding=True)
        embeddings = model.projector(output.last_hidden_state[:, 0])
    return embeddings.float()


@torch.inference_mode()
def encode_hdf5_rows(
    model,
    pixels_ds,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    rows = np.asarray(rows, dtype=np.int64)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    encoded = []
    for start in range(0, len(unique_rows), batch_size):
        batch_rows = unique_rows[start : start + batch_size]
        images = np.asarray(pixels_ds[batch_rows])
        encoded.append(encode_image_array(model, images, device).cpu())
        if start == 0 or (start // batch_size + 1) % 50 == 0:
            log(f"Encoded {min(start + batch_size, len(unique_rows))}/{len(unique_rows)} unique frames")
    unique_embeddings = torch.cat(encoded, dim=0)
    return unique_embeddings[torch.from_numpy(inverse)]


def make_episode_refs(
    episode_ids: np.ndarray,
    lengths: np.ndarray,
    rng: np.random.Generator,
    count: int,
    min_t: int,
    one_per_episode: bool = False,
) -> list[tuple[int, int]]:
    pools: dict[int, list[int]] = {}
    for episode in map(int, episode_ids):
        times = list(range(min_t, int(lengths[episode]) - FRAME_SKIP, FRAME_SKIP))
        rng.shuffle(times)
        if times:
            pools[episode] = times
    if one_per_episode:
        episodes = np.array(list(pools), dtype=np.int64)
        rng.shuffle(episodes)
        return [(int(ep), pools[int(ep)].pop()) for ep in episodes[:count]]

    refs: list[tuple[int, int]] = []
    active = list(pools)
    while active and len(refs) < count:
        rng.shuffle(active)
        next_active = []
        for episode in active:
            refs.append((episode, pools[episode].pop()))
            if pools[episode]:
                next_active.append(episode)
            if len(refs) == count:
                break
        active = next_active
    return refs


def rows_for_refs(refs: list[tuple[int, int]], offsets: np.ndarray, delta: int = 0) -> np.ndarray:
    return np.asarray([int(offsets[ep]) + t + delta for ep, t in refs], dtype=np.int64)


def action_blocks(
    refs: list[tuple[int, int]],
    offsets: np.ndarray,
    actions: np.ndarray,
    delta: int = 0,
) -> np.ndarray:
    blocks = []
    for episode, t in refs:
        row = int(offsets[episode]) + t + delta
        blocks.append(actions[row : row + FRAME_SKIP].reshape(-1))
    return np.asarray(blocks, dtype=np.float32)


@torch.inference_mode()
def predict_next(
    model,
    history_z: torch.Tensor,
    action_sequence: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    predictions = []
    for start in range(0, len(history_z), batch_size):
        z = history_z[start : start + batch_size]
        actions = action_sequence[start : start + batch_size]
        with torch.autocast(
            device_type=z.device.type, dtype=torch.bfloat16, enabled=z.device.type == "cuda"
        ):
            action_embeddings = model.action_encoder(actions)
            pred = model.predict(z, action_embeddings)[:, -1]
        predictions.append(pred.float())
    return torch.cat(predictions, dim=0)


@dataclass
class DescriptorSpace:
    z_mean: torch.Tensor
    z_std: torch.Tensor
    u_mean: torch.Tensor
    u_std: torch.Tensor
    dz_mean: torch.Tensor
    dz_std: torch.Tensor

    @classmethod
    def fit(cls, z: torch.Tensor, u: torch.Tensor, z_next: torch.Tensor):
        dz = z_next - z

        def stats(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return value.mean(0), value.std(0, unbiased=False).clamp_min(1e-6)

        return cls(*stats(z), *stats(u), *stats(dz))

    def z_scaled(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mean) / self.z_std / math.sqrt(z.shape[-1])

    def u_scaled(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.u_mean) / self.u_std / math.sqrt(u.shape[-1])

    def dz_scaled(self, z: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        dz = z_next - z
        return (dz - self.dz_mean) / self.dz_std / math.sqrt(dz.shape[-1])

    def context(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.z_scaled(z), self.u_scaled(u)], dim=-1)

    def transition(self, z: torch.Tensor, u: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.context(z, u), self.dz_scaled(z, z_next)], dim=-1)


@torch.inference_mode()
def knn_metrics(
    space: DescriptorSpace,
    z: torch.Tensor,
    u: torch.Tensor,
    z_next: torch.Tensor,
    bank_context: torch.Tensor,
    bank_transition: torch.Tensor,
    bank_dz: torch.Tensor,
    k: int,
    query_batch: int = 256,
) -> dict[str, np.ndarray]:
    all_metrics = {key: [] for key in ("context_distance", "novelty", "consistency", "dispersion")}
    for start in range(0, len(z), query_batch):
        z_b = z[start : start + query_batch]
        u_b = u[start : start + query_batch]
        next_b = z_next[start : start + query_batch]
        context = space.context(z_b, u_b)
        transition = space.transition(z_b, u_b, next_b)

        context_distances = torch.cdist(context, bank_context)
        context_knn, context_idx = torch.topk(context_distances, k, largest=False, dim=1)
        del context_distances
        transition_distances = torch.cdist(transition, bank_transition)
        transition_knn = torch.topk(transition_distances, k, largest=False, dim=1).values
        del transition_distances

        local_expected_dz = bank_dz[context_idx].mean(dim=1)
        actual_dz = space.dz_scaled(z_b, next_b)
        consistency = (actual_dz - local_expected_dz).pow(2).sum(dim=1).sqrt()

        all_metrics["context_distance"].append(context_knn.mean(dim=1).cpu())
        all_metrics["novelty"].append(transition_knn.mean(dim=1).cpu())
        all_metrics["consistency"].append(consistency.cpu())
        all_metrics["dispersion"].append(transition_knn.std(dim=1, unbiased=False).cpu())

    return {key: torch.cat(parts).numpy() for key, parts in all_metrics.items()}


def quantile_map(values: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_sorted = np.sort(np.asarray(source, dtype=np.float64))
    target_sorted = np.sort(np.asarray(target, dtype=np.float64))
    ranks = np.searchsorted(source_sorted, values, side="right") - 0.5
    quantiles = np.clip(ranks / max(1, len(source_sorted) - 1), 0.0, 1.0)
    target_positions = quantiles * (len(target_sorted) - 1)
    return np.interp(target_positions, np.arange(len(target_sorted)), target_sorted)


def bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float]:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    n = len(difference)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, n, size=(size, n))
        means[start : start + size] = difference[indices].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_probability_gt_zero": float((means > 0).mean()),
        "paired_win_rate": float((difference > 0).mean()),
    }


def summarize_selection(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    output = {}
    for method, rows in frame.groupby("method", sort=False):
        output[method] = {
            "states": int(len(rows)),
            "real_novelty_mean": float(rows.real_novelty.mean()),
            "real_novelty_median": float(rows.real_novelty.median()),
            "prediction_error_mean": float(rows.prediction_error.mean()),
            "prediction_error_median": float(rows.prediction_error.median()),
            "valid_novel_rate": float(rows.real_valid_novel.mean()),
            "predicted_valid_rate": float(rows.predicted_valid.mean()),
        }
    return output


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_closed_loop(args: argparse.Namespace) -> None:
    """Run paired multi-block rollouts using the same calibrated selector."""
    if not 0.0 <= args.reliability_decay < 1.0:
        raise ValueError("--reliability-decay must be in [0, 1)")
    if args.error_fallback_steps < 0:
        raise ValueError("--error-fallback-steps must be non-negative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    selection_rng = np.random.default_rng(args.seed + 1)
    bootstrap_rng = np.random.default_rng(args.seed + 2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()

    log(f"Loading official checkpoint for closed-loop validation on {device}")
    model, checkpoint_manifest = load_official_model(args.checkpoint_dir, device)

    with h5py.File(args.dataset, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as data:
        required = {"ep_len", "ep_offset", "pixels", "action", "state"}
        missing = sorted(required - set(data.keys()))
        if missing:
            raise KeyError(f"Dataset is missing required columns: {missing}; found {sorted(data.keys())}")
        lengths = np.asarray(data["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(data["ep_offset"][:], dtype=np.int64)
        actions = np.asarray(data["action"][:], dtype=np.float32)
        states = np.asarray(data["state"][:], dtype=np.float64)
        episode_goal_rows = offsets + lengths - 1
        episode_goal_states = np.asarray(data["state"][episode_goal_rows], dtype=np.float64)
        valid_actions = actions[np.isfinite(actions).all(axis=1)]
        action_mean = valid_actions.mean(axis=0, keepdims=True)
        action_std = np.maximum(valid_actions.std(axis=0, ddof=1, keepdims=True), 1e-6)
        sanitized_actions = np.where(np.isfinite(actions), actions, action_mean)
        normalized_actions = np.nan_to_num(
            (actions - action_mean) / action_std, nan=0.0, posinf=0.0, neginf=0.0
        )

        eligible_episodes = np.flatnonzero(lengths >= HISTORY * FRAME_SKIP + 1)
        rng.shuffle(eligible_episodes)
        n_episodes = len(eligible_episodes)
        n_train = max(1, int(0.70 * n_episodes))
        n_calibration = max(1, int(0.15 * n_episodes))
        train_episodes = eligible_episodes[:n_train]
        calibration_episodes = eligible_episodes[n_train : n_train + n_calibration]
        test_episodes = eligible_episodes[n_train + n_calibration :]
        bank_refs = make_episode_refs(train_episodes, lengths, rng, args.bank_size, min_t=0)
        calibration_refs = make_episode_refs(
            calibration_episodes, lengths, rng, args.calibration_size, min_t=10
        )
        test_refs = make_episode_refs(
            test_episodes, lengths, rng, args.test_states, min_t=10, one_per_episode=True
        )
        if len(bank_refs) < args.bank_size:
            log(f"Requested {args.bank_size} bank transitions; using {len(bank_refs)}")
        if len(calibration_refs) < max(100, args.calibration_size // 2):
            raise RuntimeError(f"Only {len(calibration_refs)} calibration transitions are available")
        if len(test_refs) < args.test_states:
            raise RuntimeError(f"Only {len(test_refs)} independent test states are available")
        ref_sets = [set(bank_refs), set(calibration_refs), set(test_refs)]
        if any(len(s) != len(r) for s, r in zip(ref_sets, (bank_refs, calibration_refs, test_refs))):
            raise RuntimeError("Duplicate transition reference detected")
        if (ref_sets[0] & ref_sets[1]) or (ref_sets[0] & ref_sets[2]) or (ref_sets[1] & ref_sets[2]):
            raise RuntimeError("Episode-step split overlap detected")

        bank_start_rows = rows_for_refs(bank_refs, offsets)
        bank_end_rows = rows_for_refs(bank_refs, offsets, FRAME_SKIP)
        bank_embeddings = encode_hdf5_rows(
            model,
            data["pixels"],
            np.concatenate([bank_start_rows, bank_end_rows]),
            device,
            args.encode_batch,
        ).to(device)
        bank_z = bank_embeddings[: len(bank_refs)]
        bank_z_next = bank_embeddings[len(bank_refs) :]
        bank_u_raw = action_blocks(bank_refs, offsets, sanitized_actions)
        bank_u = torch.from_numpy(action_blocks(bank_refs, offsets, normalized_actions)).to(device)
        space = DescriptorSpace.fit(bank_z, bank_u, bank_z_next)
        bank_context = space.context(bank_z, bank_u).contiguous()
        bank_transition = space.transition(bank_z, bank_u, bank_z_next).contiguous()
        bank_dz = space.dz_scaled(bank_z, bank_z_next).contiguous()
        coverage_threshold = float(
            torch.linalg.vector_norm(
                space.z_scaled(bank_z_next) - space.z_scaled(bank_z), dim=1
            ).median().item()
        )

        calibration_rows = np.stack(
            [rows_for_refs(calibration_refs, offsets, delta) for delta in (-10, -5, 0, 5)], axis=1
        )
        calibration_z = encode_hdf5_rows(
            model, data["pixels"], calibration_rows.reshape(-1), device, args.encode_batch
        ).reshape(len(calibration_refs), 4, -1).to(device)
        calibration_action = np.stack(
            [
                action_blocks(calibration_refs, offsets, normalized_actions, delta)
                for delta in (-10, -5, 0)
            ],
            axis=1,
        )
        calibration_action_t = torch.from_numpy(calibration_action).to(device)
        calibration_pred = predict_next(model, calibration_z[:, :3], calibration_action_t)
        calibration_actual = calibration_z[:, 3]
        calibration_current = calibration_z[:, 2]
        calibration_u = calibration_action_t[:, 2]
        predicted_cal_metrics = knn_metrics(
            space, calibration_current, calibration_u, calibration_pred,
            bank_context, bank_transition, bank_dz, args.knn_k,
        )
        actual_cal_metrics = knn_metrics(
            space, calibration_current, calibration_u, calibration_actual,
            bank_context, bank_transition, bank_dz, args.knn_k,
        )
        calibration_error = (
            (calibration_pred - calibration_actual).pow(2).mean(dim=1).cpu().numpy()
        )
        context_q95 = max(1e-12, float(np.quantile(predicted_cal_metrics["context_distance"], 0.95)))
        consistency_q95 = max(1e-12, float(np.quantile(predicted_cal_metrics["consistency"], 0.95)))
        calibration_risk = np.maximum(
            predicted_cal_metrics["context_distance"] / context_q95,
            predicted_cal_metrics["consistency"] / consistency_q95,
        )
        joint_risk_threshold = float(np.quantile(calibration_risk, 0.95))
        novelty_threshold = float(np.quantile(actual_cal_metrics["novelty"], 0.95))
        error_threshold = float(np.quantile(calibration_error, 0.95))
        dispersion_q95 = max(1e-12, float(np.quantile(predicted_cal_metrics["dispersion"], 0.95)))
        novelty_scale = max(1e-12, float(np.std(actual_cal_metrics["novelty"])))
        log(
            f"Closed-loop calibration: joint coverage={(calibration_risk <= joint_risk_threshold).mean():.3f}, "
            f"novelty_q95={novelty_threshold:.6f}, error_q95={error_threshold:.6f}"
        )

        test_rows = np.stack(
            [rows_for_refs(test_refs, offsets, delta) for delta in (-10, -5, 0)], axis=1
        )
        test_z = encode_hdf5_rows(
            model, data["pixels"], test_rows.reshape(-1), device, args.encode_batch
        ).reshape(len(test_refs), 3, -1).to(device)
        methods = ("random", "novelty_only", "frontier_static", "frontier")
        envs = {
            method: gym.make("swm/PushT-v1", resolution=224, render_mode="rgb_array")
            for method in methods
        }
        decision_records: list[dict] = []
        episode_records: list[dict] = []
        try:
            for state_id, (episode, t) in enumerate(test_refs):
                row = int(offsets[episode]) + t
                initial_state = states[row].copy()
                goal_state = episode_goal_states[episode].copy()
                seed_value = args.seed + int(episode)
                source_pool = [
                    rng.choice(len(bank_refs), size=args.candidates, replace=False)
                    for _ in range(args.rollout_blocks)
                ]
                random_choices = [int(selection_rng.integers(args.candidates)) for _ in range(args.rollout_blocks)]
                for method in methods:
                    env = envs[method]
                    env.reset(
                        seed=seed_value,
                        options={"state": initial_state.copy(), "goal_state": goal_state.copy()},
                    )
                    current_pixels = np.asarray(env.render()).copy()
                    history_latents = test_z[state_id].clone()
                    past_blocks = [
                        torch.from_numpy(
                            action_blocks([(episode, t)], offsets, normalized_actions, delta)[0]
                        ).to(device)
                        for delta in (-10, -5)
                    ]
                    episode_novelty = []
                    episode_errors = []
                    episode_valid = []
                    episode_rewards = []
                    selection_times = []
                    environment_times = []
                    evaluation_times = []
                    fallback_flags = []
                    error_exceedance_flags = []
                    visited_z = [space.z_scaled(history_latents[-1:].clone())[0].cpu()]
                    reliability_inflation = 1.0
                    fallback_remaining = 0
                    terminated = False
                    for decision in range(args.rollout_blocks):
                        scoring_started = time.perf_counter()
                        source_indices = source_pool[decision]
                        source_t = torch.from_numpy(source_indices).to(device)
                        candidate_u = bank_u[source_t]
                        candidate_raw = bank_u_raw[source_indices].reshape(args.candidates, FRAME_SKIP, -1)
                        history_batch = history_latents.unsqueeze(0).expand(args.candidates, -1, -1)
                        past_batch = torch.stack(past_blocks).unsqueeze(0).expand(args.candidates, -1, -1)
                        action_sequence = torch.cat([past_batch, candidate_u.unsqueeze(1)], dim=1)
                        predicted_next = predict_next(model, history_batch, action_sequence)
                        current_z = history_latents[-1].unsqueeze(0).expand(args.candidates, -1)
                        predicted_metrics = knn_metrics(
                            space, current_z, candidate_u, predicted_next,
                            bank_context, bank_transition, bank_dz, args.knn_k,
                        )
                        calibrated_novelty = quantile_map(
                            predicted_metrics["novelty"],
                            predicted_cal_metrics["novelty"],
                            actual_cal_metrics["novelty"],
                        )
                        joint_risk = np.maximum(
                            predicted_metrics["context_distance"] / context_q95,
                            predicted_metrics["consistency"] / consistency_q95,
                        )
                        predicted_valid = joint_risk <= joint_risk_threshold
                        frontier_score = calibrated_novelty + (
                            0.10 * novelty_scale * predicted_metrics["dispersion"] / dispersion_q95
                        )
                        reliability_before = reliability_inflation
                        adaptive_joint_risk = joint_risk * reliability_before
                        adaptive_valid = adaptive_joint_risk <= joint_risk_threshold
                        used_error_fallback = False
                        if method == "random":
                            selected_idx = random_choices[decision]
                            selection_mode = "random"
                        elif method == "novelty_only":
                            selected_idx = int(np.argmax(calibrated_novelty))
                            selection_mode = "novelty_only"
                        elif method == "frontier_static" and predicted_valid.any():
                            valid_indices = np.flatnonzero(predicted_valid)
                            selected_idx = int(valid_indices[np.argmax(frontier_score[valid_indices])])
                            selection_mode = "frontier_static"
                        elif method == "frontier_static":
                            selected_idx = int(np.argmin(joint_risk))
                            selection_mode = "risk_fallback"
                        elif fallback_remaining > 0:
                            selected_idx = int(np.argmin(adaptive_joint_risk))
                            selection_mode = "error_fallback"
                            used_error_fallback = True
                            fallback_remaining -= 1
                        elif adaptive_valid.any():
                            valid_indices = np.flatnonzero(adaptive_valid)
                            selected_idx = int(valid_indices[np.argmax(frontier_score[valid_indices])])
                            selection_mode = "frontier_adaptive"
                        else:
                            selected_idx = int(np.argmin(adaptive_joint_risk))
                            selection_mode = "risk_fallback"
                        scoring_seconds = time.perf_counter() - scoring_started

                        start_pixels = current_pixels.copy()
                        block_reward = 0.0
                        executed = 0
                        environment_started = time.perf_counter()
                        for raw_action in candidate_raw[selected_idx]:
                            _, reward, done, truncated, _ = env.step(raw_action.astype(np.float32))
                            block_reward += float(reward)
                            executed += 1
                            terminated = bool(done or truncated)
                            if terminated:
                                break
                        current_pixels = np.asarray(env.render()).copy()
                        environment_seconds = time.perf_counter() - environment_started
                        evaluation_started = time.perf_counter()
                        actual_next = encode_image_array(model, current_pixels[None], device)[0]
                        selected_u = candidate_u[selected_idx]
                        actual_metrics = knn_metrics(
                            space,
                            history_latents[-1].unsqueeze(0),
                            selected_u.unsqueeze(0),
                            actual_next.unsqueeze(0),
                            bank_context, bank_transition, bank_dz, args.knn_k,
                        )
                        prediction_error = float(
                            (predicted_next[selected_idx] - actual_next).pow(2).mean().item()
                        )
                        real_novelty = float(actual_metrics["novelty"][0])
                        real_valid_novel = bool(
                            real_novelty >= novelty_threshold and prediction_error <= error_threshold
                        )
                        error_ratio = prediction_error / max(error_threshold, 1e-12)
                        error_exceeded = error_ratio > 1.0
                        if method == "frontier":
                            reliability_inflation = max(
                                1.0,
                                args.reliability_decay * reliability_before
                                + (1.0 - args.reliability_decay) * max(1.0, error_ratio),
                            )
                            if error_exceeded:
                                fallback_remaining = max(
                                    fallback_remaining, args.error_fallback_steps
                                )
                        evaluation_seconds = time.perf_counter() - evaluation_started
                        method_predicted_valid = (
                            bool(adaptive_valid[selected_idx])
                            if method == "frontier"
                            else bool(predicted_valid[selected_idx])
                        )
                        decision_records.append(
                            {
                                "state_id": state_id,
                                "episode": int(episode),
                                "start_step": int(t),
                                "method": method,
                                "decision": decision,
                                "candidate": selected_idx,
                                "candidate_source_bank_index": int(source_indices[selected_idx]),
                                "executed_raw_actions": executed,
                                "predicted_novelty": float(predicted_metrics["novelty"][selected_idx]),
                                "predicted_novelty_calibrated": float(calibrated_novelty[selected_idx]),
                                "predicted_context_distance": float(predicted_metrics["context_distance"][selected_idx]),
                                "predicted_consistency": float(predicted_metrics["consistency"][selected_idx]),
                                "predicted_dispersion": float(predicted_metrics["dispersion"][selected_idx]),
                                "joint_risk": float(joint_risk[selected_idx]),
                                "adaptive_joint_risk": float(adaptive_joint_risk[selected_idx]),
                                "predicted_error_upper": float(
                                    error_threshold
                                    * adaptive_joint_risk[selected_idx]
                                    / max(joint_risk_threshold, 1e-12)
                                ),
                                "predicted_valid": method_predicted_valid,
                                "selection_mode": selection_mode,
                                "reliability_inflation_before": reliability_before,
                                "reliability_inflation_after": reliability_inflation,
                                "used_error_fallback": used_error_fallback,
                                "real_novelty": real_novelty,
                                "prediction_error": prediction_error,
                                "prediction_error_ratio": error_ratio,
                                "prediction_error_exceeded_q95": error_exceeded,
                                "real_valid_novel": real_valid_novel,
                                "block_reward": block_reward,
                                "terminated": terminated,
                                "selection_seconds": 0.0 if method == "random" else scoring_seconds,
                                "candidate_evaluation_seconds": scoring_seconds,
                                "environment_seconds": environment_seconds,
                                "metric_evaluation_seconds": evaluation_seconds,
                            }
                        )
                        episode_novelty.append(real_novelty)
                        episode_errors.append(prediction_error)
                        episode_valid.append(real_valid_novel)
                        episode_rewards.append(block_reward)
                        selection_times.append(0.0 if method == "random" else scoring_seconds)
                        environment_times.append(environment_seconds)
                        evaluation_times.append(evaluation_seconds)
                        fallback_flags.append(used_error_fallback)
                        error_exceedance_flags.append(error_exceeded)
                        visited_z.append(space.z_scaled(actual_next.unsqueeze(0))[0].cpu())
                        history_latents = torch.cat([history_latents[1:], actual_next.unsqueeze(0)], dim=0)
                        past_blocks = [past_blocks[-1], selected_u.detach()]
                        if terminated:
                            break
                    visited = torch.stack(visited_z)
                    path_length = float(
                        torch.linalg.vector_norm(visited[1:] - visited[:-1], dim=1).sum().item()
                    )
                    coverage_radius = float(
                        torch.linalg.vector_norm(visited - visited[:1], dim=1).max().item()
                    )
                    pairwise_spread = float(torch.pdist(visited).mean().item()) if len(visited) > 1 else 0.0
                    unique_visited = [visited[0]]
                    for latent in visited[1:]:
                        distances = torch.linalg.vector_norm(
                            torch.stack(unique_visited) - latent.unsqueeze(0), dim=1
                        )
                        if float(distances.min()) > coverage_threshold:
                            unique_visited.append(latent)
                    episode_records.append(
                        {
                            "state_id": state_id,
                            "episode": int(episode),
                            "start_step": int(t),
                            "method": method,
                            "decisions": len(episode_novelty),
                            "cumulative_real_novelty": float(np.sum(episode_novelty)),
                            "mean_real_novelty": float(np.mean(episode_novelty)) if episode_novelty else float("nan"),
                            "final_real_novelty": float(episode_novelty[-1]) if episode_novelty else float("nan"),
                            "mean_prediction_error": float(np.mean(episode_errors)) if episode_errors else float("nan"),
                            "error_exceedance_rate": float(np.mean(error_exceedance_flags)) if error_exceedance_flags else 0.0,
                            "valid_novel_rate": float(np.mean(episode_valid)) if episode_valid else 0.0,
                            "fallback_rate": float(np.mean(fallback_flags)) if fallback_flags else 0.0,
                            "total_reward": float(np.sum(episode_rewards)),
                            "terminated": bool(terminated),
                            "latent_path_length": path_length,
                            "latent_coverage_radius": coverage_radius,
                            "latent_pairwise_spread": pairwise_spread,
                            "unique_latent_states": len(unique_visited),
                            "selection_seconds_mean": float(np.mean(selection_times)) if selection_times else 0.0,
                            "environment_seconds_mean": float(np.mean(environment_times)) if environment_times else 0.0,
                            "metric_evaluation_seconds_mean": float(np.mean(evaluation_times)) if evaluation_times else 0.0,
                        }
                    )
                if (state_id + 1) % 10 == 0 or state_id == 0:
                    log(f"Closed-loop rollout {state_id + 1}/{len(test_refs)} paired states")
        finally:
            for env in envs.values():
                env.close()

    decision_frame = pd.DataFrame(decision_records)
    episode_frame = pd.DataFrame(episode_records)
    decision_frame.to_csv(args.output / "closed_loop_decisions.csv", index=False)
    episode_frame.to_csv(args.output / "closed_loop_episodes.csv", index=False)
    episode_pivots = {
        metric: episode_frame.pivot(index="state_id", columns="method", values=metric)
        for metric in (
            "cumulative_real_novelty",
            "mean_real_novelty",
            "final_real_novelty",
            "valid_novel_rate",
            "mean_prediction_error",
            "error_exceedance_rate",
            "total_reward",
            "latent_path_length",
            "latent_coverage_radius",
            "latent_pairwise_spread",
            "unique_latent_states",
        )
    }
    paired_metrics = (
        "cumulative_real_novelty", "mean_real_novelty", "final_real_novelty",
        "valid_novel_rate", "mean_prediction_error", "error_exceedance_rate", "total_reward",
        "latent_path_length", "latent_coverage_radius", "latent_pairwise_spread",
        "unique_latent_states",
    )

    def paired_against(reference: str) -> dict[str, dict[str, float]]:
        return {
            metric: bootstrap_difference(
                episode_pivots[metric].frontier.to_numpy(),
                episode_pivots[metric][reference].to_numpy(),
                bootstrap_rng,
                args.bootstrap_samples,
            )
            for metric in paired_metrics
        }

    paired = paired_against("random")
    paired_static = paired_against("frontier_static")
    method_summary = {}
    for method, rows in episode_frame.groupby("method", sort=False):
        method_decisions = decision_frame[decision_frame.method == method]
        method_summary[method] = {
            "states": int(len(rows)),
            "mean_decisions": float(rows.decisions.mean()),
            "cumulative_real_novelty_mean": float(rows.cumulative_real_novelty.mean()),
            "real_novelty_per_decision_mean": float(rows.mean_real_novelty.mean()),
            "final_real_novelty_mean": float(rows.final_real_novelty.mean()),
            "valid_novel_rate_mean": float(rows.valid_novel_rate.mean()),
            "prediction_error_mean": float(rows.mean_prediction_error.mean()),
            "error_exceedance_rate_mean": float(rows.error_exceedance_rate.mean()),
            "total_reward_mean": float(rows.total_reward.mean()),
            "success_termination_rate": float(rows.terminated.mean()),
            "latent_path_length_mean": float(rows.latent_path_length.mean()),
            "latent_coverage_radius_mean": float(rows.latent_coverage_radius.mean()),
            "latent_pairwise_spread_mean": float(rows.latent_pairwise_spread.mean()),
            "unique_latent_states_mean": float(rows.unique_latent_states.mean()),
            "fallback_rate_mean": float(rows.fallback_rate.mean()),
            "reliability_inflation_mean": float(
                method_decisions.reliability_inflation_before.mean()
            ),
            "selection_seconds_per_decision": float(rows.selection_seconds_mean.mean()),
            "environment_seconds_per_block": float(rows.environment_seconds_mean.mean()),
            "metric_evaluation_seconds_per_decision": float(rows.metric_evaluation_seconds_mean.mean()),
        }
    by_decision = []
    for (decision, method), rows in decision_frame.groupby(["decision", "method"], sort=True):
        by_decision.append(
            {
                "decision": int(decision),
                "method": method,
                "active_states": int(len(rows)),
                "real_novelty_mean": float(rows.real_novelty.mean()),
                "prediction_error_mean": float(rows.prediction_error.mean()),
                "error_exceedance_rate": float(rows.prediction_error_exceeded_q95.mean()),
                "valid_novel_rate": float(rows.real_valid_novel.mean()),
                "predicted_valid_rate": float(rows.predicted_valid.mean()),
                "error_fallback_rate": float(rows.used_error_fallback.mean()),
                "reliability_inflation_mean": float(rows.reliability_inflation_before.mean()),
                "block_reward_mean": float(rows.block_reward.mean()),
                "success_termination_rate": float(rows.terminated.mean()),
                "selection_seconds_mean": float(rows.selection_seconds.mean()),
            }
        )
    summary = {
        "experiment": {
            "type": "paired closed-loop rollout",
            "methods": list(methods),
            "test_states": len(test_refs),
            "rollout_blocks": args.rollout_blocks,
            "raw_actions_per_block": FRAME_SKIP,
            "candidate_pool_per_decision": args.candidates,
            "same_candidate_pool_across_methods": True,
            "online_bank_update": False,
            "frontier_static_is_original_selector": True,
            "frontier_uses_only_past_realized_errors": True,
            "latent_unique_state_threshold": coverage_threshold,
            "latent_unique_state_threshold_definition": "median standardized bank transition displacement",
        },
        "methods": method_summary,
        "by_decision": by_decision,
        "paired_frontier_minus_random": paired,
        "paired_frontier_minus_static": paired_static,
        "calibration": {
            "samples": len(calibration_refs),
            "context_q95": context_q95,
            "consistency_q95": consistency_q95,
            "joint_risk_threshold": joint_risk_threshold,
            "joint_valid_coverage": float((calibration_risk <= joint_risk_threshold).mean()),
            "real_novelty_q95": novelty_threshold,
            "prediction_error_q95": error_threshold,
        },
        "rollout_reliability_control": {
            "error_limit": "held-out one-step prediction error q95",
            "reliability_decay": args.reliability_decay,
            "error_fallback_steps": args.error_fallback_steps,
            "risk_update": "joint_risk * EMA(max(1, realized_error / error_q95))",
            "fallback": "minimum adjusted-risk candidate after an error-limit violation",
        },
    }
    summary["decision"] = {
        "frontier_beats_random_on_cumulative_real_novelty": bool(
            episode_pivots["cumulative_real_novelty"].frontier.mean()
            > episode_pivots["cumulative_real_novelty"].random.mean()
        ),
        "frontier_beats_random_on_valid_novel_rate": bool(
            episode_pivots["valid_novel_rate"].frontier.mean()
            > episode_pivots["valid_novel_rate"].random.mean()
        ),
        "frontier_reduces_error_vs_static": bool(
            episode_pivots["mean_prediction_error"].frontier.mean()
            < episode_pivots["mean_prediction_error"].frontier_static.mean()
        ),
        "frontier_retains_novelty_gain_vs_random": bool(
            paired["cumulative_real_novelty"]["ci95_low"] > 0
        ),
        "novelty_ci_excludes_zero": bool(
            paired["cumulative_real_novelty"]["ci95_low"] > 0
            or paired["cumulative_real_novelty"]["ci95_high"] < 0
        ),
    }
    manifest = {
        "script": Path(__file__).name,
        "mode": "closed_loop",
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
        "dataset_bytes": args.dataset.stat().st_size,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint": checkpoint_manifest,
        "split": {
            "eligible_episodes": int(n_episodes),
            "train_episodes": train_episodes.tolist(),
            "calibration_episodes": calibration_episodes.tolist(),
            "test_episodes": test_episodes.tolist(),
            "bank_transitions": len(bank_refs),
            "calibration_transitions": len(calibration_refs),
            "test_states": len(test_refs),
            "bank_duplicate_refs": len(bank_refs) - len(set(bank_refs)),
            "split_ref_overlap": 0,
        },
        "action_normalization": {
            "mean": action_mean.reshape(-1).tolist(),
            "std_unbiased": action_std.reshape(-1).tolist(),
        },
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(json_ready(summary), stream, indent=2, ensure_ascii=True, allow_nan=False)
    with (args.output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(json_ready(manifest), stream, indent=2, ensure_ascii=True, allow_nan=False)
    log(f"Closed-loop finished in {manifest['runtime']['seconds'] / 60:.1f} minutes")
    log(json.dumps(paired["cumulative_real_novelty"], indent=2))
    log(f"Structured results: {args.output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bank-size", type=int, default=20_000)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--test-states", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--rollout-blocks", type=int, default=10)
    parser.add_argument("--reliability-decay", type=float, default=0.5)
    parser.add_argument("--error-fallback-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.closed_loop:
        run_closed_loop(args)
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    selection_rng = np.random.default_rng(args.seed + 1)
    bootstrap_rng = np.random.default_rng(args.seed + 2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()

    log(f"Loading official checkpoint on {device}")
    model, checkpoint_manifest = load_official_model(args.checkpoint_dir, device)
    log(
        f"Strict checkpoint load succeeded: {checkpoint_manifest['parameters']:,} parameters, "
        f"sha256={checkpoint_manifest['weights_sha256']}"
    )

    with h5py.File(args.dataset, "r", swmr=True, rdcc_nbytes=512 * 1024 * 1024) as data:
        required = {"ep_len", "ep_offset", "pixels", "action", "state"}
        missing = sorted(required - set(data.keys()))
        if missing:
            raise KeyError(f"Dataset is missing required columns: {missing}; found {sorted(data.keys())}")

        lengths = np.asarray(data["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(data["ep_offset"][:], dtype=np.int64)
        actions = np.asarray(data["action"][:], dtype=np.float32)
        states = np.asarray(data["state"][:], dtype=np.float64)
        episode_goal_rows = offsets + lengths - 1
        episode_goal_states = np.asarray(data["state"][episode_goal_rows], dtype=np.float64)
        seed_column = np.asarray(data["seed"][:]) if "seed" in data else None

        valid_actions = actions[np.isfinite(actions).all(axis=1)]
        action_mean = valid_actions.mean(axis=0, keepdims=True)
        action_std = valid_actions.std(axis=0, ddof=1, keepdims=True)
        action_std = np.maximum(action_std, 1e-6)
        sanitized_actions = np.where(np.isfinite(actions), actions, action_mean)
        normalized_actions = np.nan_to_num(
            (actions - action_mean) / action_std, nan=0.0, posinf=0.0, neginf=0.0
        )

        eligible_episodes = np.flatnonzero(lengths >= HISTORY * FRAME_SKIP + 1)
        rng.shuffle(eligible_episodes)
        n_episodes = len(eligible_episodes)
        n_train = max(1, int(0.70 * n_episodes))
        n_calibration = max(1, int(0.15 * n_episodes))
        if n_train + n_calibration >= n_episodes:
            raise RuntimeError(f"Not enough eligible episodes for a strict split: {n_episodes}")
        train_episodes = eligible_episodes[:n_train]
        calibration_episodes = eligible_episodes[n_train : n_train + n_calibration]
        test_episodes = eligible_episodes[n_train + n_calibration :]

        bank_refs = make_episode_refs(train_episodes, lengths, rng, args.bank_size, min_t=0)
        calibration_refs = make_episode_refs(
            calibration_episodes, lengths, rng, args.calibration_size, min_t=10
        )
        test_refs = make_episode_refs(
            test_episodes, lengths, rng, args.test_states, min_t=10, one_per_episode=True
        )
        if len(bank_refs) < args.bank_size:
            log(f"Requested {args.bank_size} bank transitions; dataset provides {len(bank_refs)}")
        if len(calibration_refs) < max(100, args.calibration_size // 2):
            raise RuntimeError(f"Only {len(calibration_refs)} calibration transitions are available")
        if len(test_refs) < args.test_states:
            raise RuntimeError(
                f"Only {len(test_refs)} independent test episodes are available; requested {args.test_states}"
            )

        all_ref_keys = [set(bank_refs), set(calibration_refs), set(test_refs)]
        if any(len(keys) != len(refs) for keys, refs in zip(all_ref_keys, [bank_refs, calibration_refs, test_refs])):
            raise RuntimeError("Duplicate (episode, step) transition detected")
        if (all_ref_keys[0] & all_ref_keys[1]) or (all_ref_keys[0] & all_ref_keys[2]) or (all_ref_keys[1] & all_ref_keys[2]):
            raise RuntimeError("Split overlap detected")
        log(
            f"Episode split train/calibration/test={len(train_episodes)}/"
            f"{len(calibration_episodes)}/{len(test_episodes)}; refs="
            f"{len(bank_refs)}/{len(calibration_refs)}/{len(test_refs)}"
        )

        bank_start_rows = rows_for_refs(bank_refs, offsets)
        bank_end_rows = rows_for_refs(bank_refs, offsets, FRAME_SKIP)
        bank_rows = np.concatenate([bank_start_rows, bank_end_rows])
        log("Encoding deduplicated bank frames")
        bank_embeddings = encode_hdf5_rows(
            model, data["pixels"], bank_rows, device, args.encode_batch
        ).to(device)
        bank_z = bank_embeddings[: len(bank_refs)]
        bank_z_next = bank_embeddings[len(bank_refs) :]
        bank_u_raw = action_blocks(bank_refs, offsets, sanitized_actions)
        bank_u = torch.from_numpy(action_blocks(bank_refs, offsets, normalized_actions)).to(device)

        space = DescriptorSpace.fit(bank_z, bank_u, bank_z_next)
        bank_context = space.context(bank_z, bank_u).contiguous()
        bank_transition = space.transition(bank_z, bank_u, bank_z_next).contiguous()
        bank_dz = space.dz_scaled(bank_z, bank_z_next).contiguous()
        if args.knn_k >= len(bank_refs):
            raise ValueError("knn-k must be smaller than the bank")
        log(
            f"KNN bank resident on {bank_context.device}: context={tuple(bank_context.shape)}, "
            f"transition={tuple(bank_transition.shape)}"
        )

        eval_refs = calibration_refs + test_refs
        eval_rows = []
        for delta in (-10, -5, 0, 5):
            eval_rows.append(rows_for_refs(eval_refs, offsets, delta))
        stacked_eval_rows = np.stack(eval_rows, axis=1)
        log("Encoding held-out calibration and test histories")
        eval_embeddings = encode_hdf5_rows(
            model, data["pixels"], stacked_eval_rows.reshape(-1), device, args.encode_batch
        ).reshape(len(eval_refs), 4, -1).to(device)

        n_cal = len(calibration_refs)
        calibration_z = eval_embeddings[:n_cal]
        calibration_action = np.stack(
            [
                action_blocks(calibration_refs, offsets, normalized_actions, delta)
                for delta in (-10, -5, 0)
            ],
            axis=1,
        )
        calibration_action_t = torch.from_numpy(calibration_action).to(device)
        calibration_pred = predict_next(
            model, calibration_z[:, :3], calibration_action_t
        )
        calibration_actual = calibration_z[:, 3]
        calibration_current = calibration_z[:, 2]
        calibration_u = calibration_action_t[:, 2]
        predicted_cal_metrics = knn_metrics(
            space,
            calibration_current,
            calibration_u,
            calibration_pred,
            bank_context,
            bank_transition,
            bank_dz,
            args.knn_k,
        )
        actual_cal_metrics = knn_metrics(
            space,
            calibration_current,
            calibration_u,
            calibration_actual,
            bank_context,
            bank_transition,
            bank_dz,
            args.knn_k,
        )
        calibration_error = (
            (calibration_pred - calibration_actual).pow(2).mean(dim=1).cpu().numpy()
        )

        context_q95 = max(1e-12, float(np.quantile(predicted_cal_metrics["context_distance"], 0.95)))
        consistency_q95 = max(1e-12, float(np.quantile(predicted_cal_metrics["consistency"], 0.95)))
        calibration_risk = np.maximum(
            predicted_cal_metrics["context_distance"] / context_q95,
            predicted_cal_metrics["consistency"] / consistency_q95,
        )
        joint_risk_threshold = float(np.quantile(calibration_risk, 0.95))
        novelty_threshold = float(np.quantile(actual_cal_metrics["novelty"], 0.95))
        error_threshold = float(np.quantile(calibration_error, 0.95))
        dispersion_q95 = max(1e-12, float(np.quantile(predicted_cal_metrics["dispersion"], 0.95)))
        novelty_scale = max(1e-12, float(np.std(actual_cal_metrics["novelty"])))
        log(
            f"Calibration complete: joint-valid coverage="
            f"{float((calibration_risk <= joint_risk_threshold).mean()):.3f}, "
            f"real novelty q95={novelty_threshold:.6f}, error q95={error_threshold:.6f}"
        )

        calibration_frame = pd.DataFrame(
            {
                "episode": [ep for ep, _ in calibration_refs],
                "step": [t for _, t in calibration_refs],
                "predicted_novelty_raw": predicted_cal_metrics["novelty"],
                "real_novelty": actual_cal_metrics["novelty"],
                "context_distance": predicted_cal_metrics["context_distance"],
                "consistency": predicted_cal_metrics["consistency"],
                "dispersion": predicted_cal_metrics["dispersion"],
                "joint_risk": calibration_risk,
                "prediction_error": calibration_error,
            }
        )
        calibration_frame["predicted_novelty_calibrated"] = quantile_map(
            calibration_frame.predicted_novelty_raw.to_numpy(),
            predicted_cal_metrics["novelty"],
            actual_cal_metrics["novelty"],
        )

        test_z = eval_embeddings[n_cal:]
        test_past_actions = np.stack(
            [
                action_blocks(test_refs, offsets, normalized_actions, delta)
                for delta in (-10, -5)
            ],
            axis=1,
        )
        test_current_rows = rows_for_refs(test_refs, offsets)
        test_pixels = np.asarray(data["pixels"][np.sort(test_current_rows)])
        sorted_row_to_pixel = {
            int(row): pixel for row, pixel in zip(np.sort(test_current_rows), test_pixels)
        }

        log(f"Creating {args.candidates} independent PushT simulators")
        envs = [
            gym.make("swm/PushT-v1", resolution=224, render_mode="rgb_array")
            for _ in range(args.candidates)
        ]
        candidate_records: list[dict] = []
        selection_records: list[dict] = []
        reconstruction_mae = []
        try:
            for state_idx, (episode, t) in enumerate(test_refs):
                candidate_source = rng.choice(len(bank_refs), size=args.candidates, replace=False)
                candidate_raw = bank_u_raw[candidate_source].reshape(args.candidates, FRAME_SKIP, -1)
                candidate_normalized = bank_u[candidate_source]

                history_z = test_z[state_idx, :3].unsqueeze(0).expand(args.candidates, -1, -1)
                past = torch.from_numpy(test_past_actions[state_idx]).to(device)
                full_actions = torch.empty(
                    (args.candidates, HISTORY, candidate_normalized.shape[-1]),
                    device=device,
                    dtype=torch.float32,
                )
                full_actions[:, :2] = past.unsqueeze(0)
                full_actions[:, 2] = candidate_normalized
                predicted_next = predict_next(model, history_z, full_actions)
                current_z = test_z[state_idx, 2].unsqueeze(0).expand(args.candidates, -1)
                predicted_metrics = knn_metrics(
                    space,
                    current_z,
                    candidate_normalized,
                    predicted_next,
                    bank_context,
                    bank_transition,
                    bank_dz,
                    args.knn_k,
                )
                predicted_calibrated = quantile_map(
                    predicted_metrics["novelty"],
                    predicted_cal_metrics["novelty"],
                    actual_cal_metrics["novelty"],
                )
                joint_risk = np.maximum(
                    predicted_metrics["context_distance"] / context_q95,
                    predicted_metrics["consistency"] / consistency_q95,
                )
                predicted_valid = joint_risk <= joint_risk_threshold
                frontier_score = predicted_calibrated + (
                    0.10
                    * novelty_scale
                    * predicted_metrics["dispersion"]
                    / dispersion_q95
                )

                row = int(offsets[episode]) + t
                state = states[row].copy()
                # The official HDF5 stores no separate goal_state. As in the
                # dataset-driven evaluator, use a future state from the same
                # episode; the PushT visual goal pose itself is fixed by the env.
                goal_state = episode_goal_states[episode].copy()
                seed_value = (
                    int(seed_column[row])
                    if seed_column is not None and np.isfinite(seed_column[row])
                    else args.seed + int(episode)
                )
                next_images = []
                terminal_flags = []
                for candidate_idx, env in enumerate(envs):
                    env.reset(
                        seed=seed_value,
                        options={"state": state.copy(), "goal_state": goal_state.copy()},
                    )
                    if candidate_idx == 0:
                        rendered_start = np.asarray(env.render())
                        dataset_start = np.asarray(sorted_row_to_pixel[row])
                        if dataset_start.shape[0] in (1, 3) and dataset_start.ndim == 3:
                            dataset_start = np.transpose(dataset_start, (1, 2, 0))
                        reconstruction_mae.append(
                            float(np.abs(rendered_start.astype(np.float32) - dataset_start.astype(np.float32)).mean() / 255.0)
                        )
                    terminated_any = False
                    for raw_action in candidate_raw[candidate_idx]:
                        _, _, terminated, truncated, _ = env.step(raw_action.astype(np.float32))
                        terminated_any = terminated_any or bool(terminated or truncated)
                    next_images.append(np.asarray(env.render()))
                    terminal_flags.append(terminated_any)

                actual_next = encode_image_array(model, np.stack(next_images), device)
                actual_metrics = knn_metrics(
                    space,
                    current_z,
                    candidate_normalized,
                    actual_next,
                    bank_context,
                    bank_transition,
                    bank_dz,
                    args.knn_k,
                )
                prediction_error = (
                    (predicted_next - actual_next).pow(2).mean(dim=1).cpu().numpy()
                )
                real_valid_novel = (
                    (actual_metrics["novelty"] >= novelty_threshold)
                    & (prediction_error <= error_threshold)
                )

                random_idx = int(selection_rng.integers(args.candidates))
                novelty_idx = int(np.argmax(predicted_calibrated))
                if predicted_valid.any():
                    valid_indices = np.flatnonzero(predicted_valid)
                    frontier_idx = int(valid_indices[np.argmax(frontier_score[valid_indices])])
                else:
                    frontier_idx = int(np.argmin(joint_risk))
                oracle_valid = prediction_error <= error_threshold
                if oracle_valid.any():
                    oracle_indices = np.flatnonzero(oracle_valid)
                    oracle_idx = int(oracle_indices[np.argmax(actual_metrics["novelty"][oracle_indices])])
                else:
                    oracle_idx = int(np.argmax(actual_metrics["novelty"]))

                method_indices = {
                    "random": random_idx,
                    "novelty_only": novelty_idx,
                    "frontier": frontier_idx,
                    "oracle": oracle_idx,
                }
                for candidate_idx in range(args.candidates):
                    record = {
                        "state_id": state_idx,
                        "episode": int(episode),
                        "step": int(t),
                        "candidate": candidate_idx,
                        "candidate_source_bank_index": int(candidate_source[candidate_idx]),
                        "predicted_novelty_raw": float(predicted_metrics["novelty"][candidate_idx]),
                        "predicted_novelty_calibrated": float(predicted_calibrated[candidate_idx]),
                        "context_distance": float(predicted_metrics["context_distance"][candidate_idx]),
                        "consistency": float(predicted_metrics["consistency"][candidate_idx]),
                        "dispersion": float(predicted_metrics["dispersion"][candidate_idx]),
                        "joint_risk": float(joint_risk[candidate_idx]),
                        "predicted_valid": bool(predicted_valid[candidate_idx]),
                        "frontier_score": float(frontier_score[candidate_idx]),
                        "real_novelty": float(actual_metrics["novelty"][candidate_idx]),
                        "prediction_error": float(prediction_error[candidate_idx]),
                        "real_valid_novel": bool(real_valid_novel[candidate_idx]),
                        "terminated_within_block": bool(terminal_flags[candidate_idx]),
                    }
                    for action_idx, action_value in enumerate(candidate_raw[candidate_idx].reshape(-1)):
                        record[f"action_{action_idx}"] = float(action_value)
                    candidate_records.append(record)

                state_candidates = candidate_records[-args.candidates :]
                for method, candidate_idx in method_indices.items():
                    selected = dict(state_candidates[candidate_idx])
                    selected["method"] = method
                    selection_records.append(selected)

                if (state_idx + 1) % 10 == 0 or state_idx == 0:
                    log(f"Counterfactual simulator test {state_idx + 1}/{len(test_refs)} states")
        finally:
            for env in envs:
                env.close()

    candidate_frame = pd.DataFrame(candidate_records)
    selection_frame = pd.DataFrame(selection_records)
    candidate_frame.to_parquet(args.output / "candidates.parquet", index=False)
    candidate_frame.to_csv(args.output / "candidates.csv", index=False)
    selection_frame.to_csv(args.output / "selections.csv", index=False)
    calibration_frame.to_parquet(args.output / "calibration.parquet", index=False)
    calibration_frame.to_csv(args.output / "calibration.csv", index=False)

    selection_summary = summarize_selection(selection_frame)
    pivot_novelty = selection_frame.pivot(index="state_id", columns="method", values="real_novelty")
    pivot_valid = selection_frame.pivot(index="state_id", columns="method", values="real_valid_novel")
    pivot_error = selection_frame.pivot(index="state_id", columns="method", values="prediction_error")
    spearman = spearmanr(
        candidate_frame.predicted_novelty_calibrated,
        candidate_frame.real_novelty,
        nan_policy="omit",
    )
    spearman_rho = getattr(spearman, "statistic", getattr(spearman, "correlation", np.nan))

    summary = {
        "decision": {
            "frontier_beats_random_on_real_novelty": bool(
                pivot_novelty.frontier.mean() > pivot_novelty.random.mean()
            ),
            "frontier_beats_random_on_valid_novel_rate": bool(
                pivot_valid.frontier.mean() > pivot_valid.random.mean()
            ),
            "credible_if_novelty_ci_excludes_zero": None,
        },
        "methods": selection_summary,
        "paired_frontier_minus_random": {
            "real_novelty": bootstrap_difference(
                pivot_novelty.frontier.to_numpy(),
                pivot_novelty.random.to_numpy(),
                bootstrap_rng,
                args.bootstrap_samples,
            ),
            "valid_novel_rate": bootstrap_difference(
                pivot_valid.frontier.to_numpy(dtype=float),
                pivot_valid.random.to_numpy(dtype=float),
                bootstrap_rng,
                args.bootstrap_samples,
            ),
            "prediction_error": bootstrap_difference(
                pivot_error.frontier.to_numpy(),
                pivot_error.random.to_numpy(),
                bootstrap_rng,
                args.bootstrap_samples,
            ),
        },
        "frontier_regret_to_oracle": {
            "mean": float((pivot_novelty.oracle - pivot_novelty.frontier).mean()),
            "median": float((pivot_novelty.oracle - pivot_novelty.frontier).median()),
        },
        "all_candidate_rank_calibration": {
            "spearman_rho": float(spearman_rho),
            "p_value": float(spearman.pvalue),
        },
        "calibration": {
            "samples": len(calibration_frame),
            "context_q95": context_q95,
            "consistency_q95": consistency_q95,
            "joint_risk_threshold": joint_risk_threshold,
            "joint_valid_coverage": float((calibration_risk <= joint_risk_threshold).mean()),
            "real_novelty_q95": novelty_threshold,
            "prediction_error_q95": error_threshold,
            "dispersion_q95": dispersion_q95,
            "novelty_mapping": "empirical quantile map: predicted calibration distribution -> real calibration distribution",
        },
        "simulator_reconstruction": {
            "dataset_vs_reset_render_mae_0_to_1_mean": float(np.mean(reconstruction_mae)),
            "dataset_vs_reset_render_mae_0_to_1_q95": float(np.quantile(reconstruction_mae, 0.95)),
        },
    }
    novelty_ci = summary["paired_frontier_minus_random"]["real_novelty"]
    summary["decision"]["credible_if_novelty_ci_excludes_zero"] = bool(
        novelty_ci["ci95_low"] > 0 or novelty_ci["ci95_high"] < 0
    )

    split_manifest = {
        "eligible_episodes": int(n_episodes),
        "train_episodes": train_episodes.tolist(),
        "calibration_episodes": calibration_episodes.tolist(),
        "test_episodes": test_episodes.tolist(),
        "bank_transitions": len(bank_refs),
        "calibration_transitions": len(calibration_refs),
        "test_states": len(test_refs),
        "bank_duplicate_refs": len(bank_refs) - len(set(bank_refs)),
        "split_ref_overlap": 0,
    }
    manifest = {
        "script": Path(__file__).name,
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
        "dataset_bytes": args.dataset.stat().st_size,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint": checkpoint_manifest,
        "split": split_manifest,
        "experiment": {
            "model_horizon_steps": 1,
            "raw_actions_per_model_step": FRAME_SKIP,
            "history_frames": HISTORY,
            "candidates_per_state": args.candidates,
            "knn_k": args.knn_k,
            "bank_device": str(bank_context.device),
            "candidate_pool": "same fixed empirical bank-action blocks for all methods",
            "paired_methods": ["random", "novelty_only", "frontier", "oracle"],
            "simulator_goal_state": "last recorded state of the same episode",
        },
        "action_normalization": {
            "mean": action_mean.reshape(-1).tolist(),
            "std_unbiased": action_std.reshape(-1).tolist(),
        },
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }

    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(json_ready(summary), stream, indent=2, ensure_ascii=True, allow_nan=False)
    with (args.output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(json_ready(manifest), stream, indent=2, ensure_ascii=True, allow_nan=False)

    log(f"Finished in {manifest['runtime']['seconds'] / 60:.1f} minutes")
    log(json.dumps(summary["paired_frontier_minus_random"], indent=2))
    log(f"Structured results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
