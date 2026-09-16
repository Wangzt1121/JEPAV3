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


def split_episodes(dataset, fraction, seed):
    episodes = np.unique([int(ep) for ep, _ in dataset.clip_indices])
    if len(episodes) < 2 or not 0 < fraction < 1:
        raise ValueError("bank construction needs at least two episodes")
    count = min(len(episodes) - 1, max(1, int(round(len(episodes) * fraction))))
    validation = set(np.random.default_rng(seed).permutation(episodes)[:count].tolist())
    train, heldout = [], []
    for index, (episode, _) in enumerate(dataset.clip_indices):
        (heldout if int(episode) in validation else train).append(index)
    return train, heldout, sorted(validation)


def sample_unique_indices(dataset, indices, count, rng):
    """Sample unique episode/step clips before encoding or FIFO insertion."""
    unique = {}
    for index in indices:
        unique.setdefault(tuple(map(int, dataset.clip_indices[index])), index)
    values = np.asarray(list(unique.values()), dtype=np.int64)
    rng.shuffle(values)
    return values[:count].tolist()


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
    parser.add_argument("--validation-fraction", type=float, default=0.1)
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
    train_indices, heldout_indices, validation_episodes = split_episodes(
        dataset, args.validation_fraction, args.seed
    )
    rng = np.random.default_rng(args.seed)
    train_indices = sample_unique_indices(
        dataset, train_indices, args.max_size, rng
    )
    heldout_indices = sample_unique_indices(
        dataset, heldout_indices, args.calibration_size, rng
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
        "validation_episodes": validation_episodes, "action_stats": action_stats,
        "descriptor_type": "action_effect", "bank_size": len(bank),
        "sampling": "unique episode-step clips, seeded without replacement",
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
        "format_version": 1,
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
