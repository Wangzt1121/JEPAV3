"""Non-parametric memory of real state/action/effect transitions."""

from pathlib import Path

import torch
import torch.nn.functional as F


class ActionEffectMemory:
    """FIFO memory used for context support, local effects and novelty.

    The bank stores only transitions produced by the real dataset/environment.
    Stored tensors live on ``device`` (CPU is recommended during exploration).
    """

    def __init__(self, max_size=50000, knn_k=10, chunk_size=256,
                 bank_chunk_size=4096, device="cpu", metadata=None):
        self.max_size = int(max_size)
        self.knn_k = int(knn_k)
        self.chunk_size = int(chunk_size)
        self.bank_chunk_size = int(bank_chunk_size)
        self.device = torch.device(device)
        if min(self.max_size, self.knn_k, self.chunk_size, self.bank_chunk_size) < 1:
            raise ValueError("memory sizes and k must be positive")
        self.metadata = {} if metadata is None else metadata
        empty = torch.empty(0, 0, dtype=torch.float32, device=self.device)
        self.state = empty.clone()
        self.action = empty.clone()
        self.effect = empty.clone()
        self.context = empty.clone()
        self.transition = empty.clone()

    def __len__(self):
        return int(self.state.shape[0])

    @staticmethod
    def _normalize(value):
        return F.normalize(value.float(), dim=-1, eps=1e-8)

    def add(self, z_t, action_emb, delta_z):
        """Add real transitions ``(z_t, action_embedding, z_next-z_t)``."""
        if z_t.shape[:-1] != action_emb.shape[:-1] or z_t.shape != delta_z.shape:
            raise ValueError("state, action and effect shapes are not aligned")
        state = self._normalize(z_t.detach()).reshape(-1, z_t.shape[-1])
        action = self._normalize(action_emb.detach()).reshape(-1, action_emb.shape[-1])
        effect = self._normalize(delta_z.detach()).reshape(-1, delta_z.shape[-1])
        if state.shape[0] == 0:
            return
        if not torch.isfinite(torch.cat([state, action, effect], dim=-1)).all():
            raise ValueError("real transition memory contains nonfinite values")
        context = self._normalize(torch.cat([state, action], dim=-1))
        transition = self._normalize(torch.cat([state, action, effect], dim=-1))
        values = (state, action, effect, context, transition)
        values = tuple(value.to(self.device) for value in values)
        if len(self) == 0:
            self.state, self.action, self.effect, self.context, self.transition = values
        else:
            if values[0].shape[1] != self.state.shape[1] or values[1].shape[1] != self.action.shape[1]:
                raise ValueError("memory feature dimensions changed")
            self.state = torch.cat([self.state, values[0]], dim=0)
            self.action = torch.cat([self.action, values[1]], dim=0)
            self.effect = torch.cat([self.effect, values[2]], dim=0)
            self.context = torch.cat([self.context, values[3]], dim=0)
            self.transition = torch.cat([self.transition, values[4]], dim=0)
        if len(self) > self.max_size:
            keep = slice(-self.max_size, None)
            self.state = self.state[keep]
            self.action = self.action[keep]
            self.effect = self.effect[keep]
            self.context = self.context[keep]
            self.transition = self.transition[keep]

    def _knn(self, query, table):
        query = self._normalize(query.detach()).reshape(-1, query.shape[-1])
        if not len(self):
            raise ValueError("cannot query an empty memory")
        if query.shape[1] != table.shape[1]:
            raise ValueError("query and memory feature dimensions differ")
        k = min(self.knn_k, len(self))
        all_values, all_indices = [], []
        for q_start in range(0, query.shape[0], self.chunk_size):
            q = query[q_start:q_start + self.chunk_size]
            best_values = q.new_full((q.shape[0], 0), -float("inf"))
            best_indices = torch.empty((q.shape[0], 0), dtype=torch.long, device=q.device)
            for b_start in range(0, len(self), self.bank_chunk_size):
                bank = table[b_start:b_start + self.bank_chunk_size].to(q.device)
                values = q @ bank.T
                local_k = min(k, values.shape[-1])
                local_values, local_indices = values.topk(local_k, dim=-1)
                local_indices = local_indices + b_start
                merged_values = torch.cat([best_values, local_values], dim=-1)
                merged_indices = torch.cat([best_indices, local_indices], dim=-1)
                keep_values, keep_positions = merged_values.topk(
                    min(k, merged_values.shape[-1]), dim=-1
                )
                best_values = keep_values
                best_indices = merged_indices.gather(-1, keep_positions)
            all_values.append(best_values)
            all_indices.append(best_indices)
        return torch.cat(all_values), torch.cat(all_indices)

    def query(self, z_t, action_emb, delta_z):
        """Return non-parametric frontier statistics for candidate effects."""
        if z_t.shape[:-1] != action_emb.shape[:-1] or z_t.shape != delta_z.shape:
            raise ValueError("query tensors are not aligned")
        leading = z_t.shape[:-1]
        state = self._normalize(z_t).reshape(-1, z_t.shape[-1])
        action = self._normalize(action_emb).reshape(-1, action_emb.shape[-1])
        effect = self._normalize(delta_z).reshape(-1, delta_z.shape[-1])
        context_query = self._normalize(torch.cat([state, action], dim=-1))
        transition_query = self._normalize(torch.cat([state, action, effect], dim=-1))
        context_sim, indices = self._knn(context_query, self.context)
        transition_sim, _ = self._knn(transition_query, self.transition)
        neighbor_effects = self.effect[indices.cpu()].to(effect.device)
        mean_effect = self._normalize(neighbor_effects.mean(dim=1))
        local_ambiguity = (neighbor_effects - mean_effect.unsqueeze(1)).pow(2).mean(dim=-1).mean(dim=-1)
        effect_consistency = 1.0 - (effect * mean_effect).sum(dim=-1).clamp(-1.0, 1.0)
        result = {
            "context_distance": 1.0 - context_sim.mean(dim=-1),
            "local_ambiguity": local_ambiguity,
            "effect_consistency": effect_consistency,
            "novelty": 1.0 - transition_sim.mean(dim=-1),
        }
        return {key: value.reshape(leading) for key, value in result.items()}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1,
            "max_size": self.max_size,
            "knn_k": self.knn_k,
            "chunk_size": self.chunk_size,
            "bank_chunk_size": self.bank_chunk_size,
            "state": self.state.cpu(),
            "action": self.action.cpu(),
            "effect": self.effect.cpu(),
            "context": self.context.cpu(),
            "transition": self.transition.cpu(),
            "metadata": self.metadata,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported training-free frontier bank format")
        bank = cls(payload["max_size"], payload["knn_k"], payload.get("chunk_size", 256),
                    payload.get("bank_chunk_size", 4096), device=device,
                    metadata=payload.get("metadata", {}))
        for name in ("state", "action", "effect", "context", "transition"):
            setattr(bank, name, payload[name].to(bank.device))
        return bank
