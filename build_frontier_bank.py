"""Build a fixed, real-only Action-Effect Memory for training-free AEFE."""

import argparse
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader, Subset

from frontier.memory import ActionEffectMemory
from frontier.score import quantiles
from utils import ZScoreNormalizer, get_img_preprocessor


def fit_action_stats(name):
    dataset = swm.data.load_dataset(name, keys_to_load=["action"])
    action = torch.as_tensor(np.asarray(dataset.get_col_data("action"))).float()
    action = action[torch.isfinite(action).all(dim=-1)]
    if action.shape[0] < 2:
        raise ValueError("not enough finite actions for normalization")
    return {"mean": action.mean(0), "std": action.std(0).clamp_min(1e-6), "source": str(name)}


def load_dataset(name, history_size, frameskip, img_size, action_stats):
    transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", img_size=img_size),
        spt.data.transforms.WrapTorchTransform(
            ZScoreNormalizer(action_stats["mean"], action_stats["std"]),
            source="action", target="action",
        ),
    )
    return swm.data.load_dataset(
        name, num_steps=history_size + 1, frameskip=frameskip,
        keys_to_load=["pixels", "action"], keys_to_cache=["action"],
        transform=transform,
    )


def split_episodes(dataset, calibration_fraction, test_fraction, seed):
    """Create strict Bank/calibration/test episode partitions."""
    episodes = np.unique([int(ep) for ep, _ in dataset.clip_indices])
    if len(episodes) < 3:
        raise ValueError("bank construction needs at least three episodes")
    if min(calibration_fraction, test_fraction) <= 0:
        raise ValueError("calibration and test fractions must be positive")
    if calibration_fraction + test_fraction >= 1:
        raise ValueError("calibration and test fractions must sum to less than one")

    episodes = np.random.default_rng(seed).permutation(episodes)
    calibration_count = max(1, int(round(len(episodes) * calibration_fraction)))
    test_count = max(1, int(round(len(episodes) * test_fraction)))
    if calibration_count + test_count >= len(episodes):
        raise ValueError("episode split leaves no episodes for the Bank")
    calibration = set(episodes[:calibration_count].tolist())
    test = set(episodes[calibration_count:calibration_count + test_count].tolist())
    bank = set(episodes[calibration_count + test_count:].tolist())

    partitions = {"bank": [], "calibration": [], "test": []}
    for index, (episode, _) in enumerate(dataset.clip_indices):
        episode = int(episode)
        if episode in bank:
            partitions["bank"].append(index)
        elif episode in calibration:
            partitions["calibration"].append(index)
        elif episode in test:
            partitions["test"].append(index)
    return partitions, {
        "bank": sorted(bank),
        "calibration": sorted(calibration),
        "test": sorted(test),
    }


def sample_episode_balanced_nonoverlapping(dataset, indices, count, frameskip, rng):
    """Round-robin episodes and keep disjoint raw-action blocks."""
    grouped = {}
    for index in indices:
        episode, start = map(int, dataset.clip_indices[index])
        grouped.setdefault(episode, {}).setdefault(start, index)

    pools = {}
    for episode, by_start in grouped.items():
        starts = np.asarray(sorted(by_start), dtype=np.int64)
        residues = np.unique(starts % frameskip)
        residue = int(rng.choice(residues))
        selected = [by_start[int(start)] for start in starts if start % frameskip == residue]
        rng.shuffle(selected)
        if selected:
            pools[episode] = selected

    sampled = []
    active = list(pools)
    while active and len(sampled) < count:
        rng.shuffle(active)
        next_active = []
        for episode in active:
            sampled.append(pools[episode].pop())
            if pools[episode]:
                next_active.append(episode)
            if len(sampled) == count:
                break
        active = next_active
    return sampled


def move_batch(batch, device):
    action = batch["action"].clone()
    if not torch.isfinite(action[:, :-1]).all():
        raise ValueError("a real transition contains a missing action")
    action[:, -1] = torch.nan_to_num(action[:, -1])
    return {"pixels": batch["pixels"].to(device), "action": action.to(device)}


def encode_last_transition(lewm, batch, device):
    batch = move_batch(batch, device)
    return lewm.encode({key: value[:, -2:] for key, value in batch.items()})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stats-dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--calibration-fraction", "--validation-fraction",
        dest="calibration_fraction", type=float, default=0.1,
    )
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=50000)
    parser.add_argument("--calibration-size", type=int, default=4096)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.history_size, args.frameskip, args.max_size, args.calibration_size) < 1:
        parser.error("history-size, frameskip, max-size and calibration-size must be positive")

    device = torch.device(args.device)
    lewm = swm.wm.utils.load_pretrained(args.checkpoint).to(device).eval()
    lewm.requires_grad_(False)
    stats_dataset = args.stats_dataset or args.dataset
    action_stats = fit_action_stats(stats_dataset)
    dataset = load_dataset(args.dataset, args.history_size, args.frameskip,
                           args.img_size, action_stats)
    partitions, episode_split = split_episodes(
        dataset, args.calibration_fraction, args.test_fraction, args.seed
    )
    rng = np.random.default_rng(args.seed)
    train_indices = sample_episode_balanced_nonoverlapping(
        dataset, partitions["bank"], args.max_size, args.frameskip, rng
    )
    heldout_indices = sample_episode_balanced_nonoverlapping(
        dataset, partitions["calibration"], args.calibration_size,
        args.frameskip, rng,
    )
    if len(train_indices) < args.max_size:
        print(f"requested_bank_size={args.max_size} available={len(train_indices)}")
    if len(heldout_indices) < args.calibration_size:
        print(
            f"requested_calibration_size={args.calibration_size} "
            f"available={len(heldout_indices)}"
        )
    bank = ActionEffectMemory(args.max_size, args.knn_k, device="cpu")
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)
    for batch_index, raw in enumerate(DataLoader(Subset(dataset, train_indices), **loader_kwargs)):
        if args.max_batches and batch_index >= args.max_batches:
            break
        output = encode_last_transition(lewm, raw, device)
        z, action = output["emb"], output["act_emb"]
        bank.add(z[:, -2], action[:, -2], z[:, -1] - z[:, -2])
        if batch_index == 0:
            print(f"emb={tuple(z.shape)} act_emb={tuple(action.shape)}")
    if not len(bank):
        raise ValueError("no transitions were added to the memory")

    metadata = {
        "checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
        "stats_dataset": str(stats_dataset), "history_size": args.history_size,
        "frameskip": args.frameskip, "img_size": args.img_size,
        "bank_episodes": episode_split["bank"],
        "calibration_episodes": episode_split["calibration"],
        "test_episodes": episode_split["test"],
        "validation_episodes": episode_split["calibration"],
        "action_stats": action_stats,
        "descriptor_type": "action_effect", "bank_size": len(bank),
        "sampling_version": 2,
        "sampling": (
            "strict episode split; seeded episode-round-robin sampling; "
            "transition starts separated by frameskip within each episode"
        ),
        "transition_per_clip": "last only",
    }
    bank.metadata = metadata
    novelty, ambiguity, consistency, context = [], [], [], []
    for batch_index, raw in enumerate(DataLoader(Subset(dataset, heldout_indices), **loader_kwargs)):
        if args.max_batches and batch_index >= args.max_batches:
            break
        output = encode_last_transition(lewm, raw, device)
        z, action = output["emb"], output["act_emb"]
        values = bank.query(z[:, -2], action[:, -2], z[:, -1] - z[:, -2])
        novelty.append(values["novelty"].flatten().cpu())
        ambiguity.append(values["local_ambiguity"].flatten().cpu())
        consistency.append(values["effect_consistency"].flatten().cpu())
        context.append(values["context_distance"].flatten().cpu())
    if not novelty:
        raise ValueError("held-out calibration set is empty")
    novelty, ambiguity = torch.cat(novelty), torch.cat(ambiguity)
    consistency, context = torch.cat(consistency), torch.cat(context)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank.save(output_dir / "frontier_bank.pt")
    torch.save({
        "format_version": 2,
        "metadata": metadata,
        "novelty": quantiles(novelty),
        "ambiguity": quantiles(ambiguity),
        "context": quantiles(context),
        "consistency": quantiles(consistency),
        "context_q95": torch.quantile(context, 0.95).item(),
        "consistency_q95": torch.quantile(consistency, 0.95).item(),
    }, output_dir / "frontier_stats.pt")
    print(f"bank_size={len(bank)} calibration_transitions={novelty.numel()}")
    print(f"context_q95={torch.quantile(context, 0.95).item():.6f} "
          f"consistency_q95={torch.quantile(consistency, 0.95).item():.6f}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
