from __future__ import annotations

import torch
import torch.nn.functional as F

from dataset import Batch

_PAD_RANK = 10_000   # sentinel rank assigned to padding slots in the batch


def _valid_pairs(ranks: torch.Tensor, cand_mask: torch.Tensor):
    rank_i = ranks.unsqueeze(2)       # [B, K, 1]
    rank_j = ranks.unsqueeze(1)       # [B, 1, K]
    mask_i = cand_mask.unsqueeze(2).bool()
    mask_j = cand_mask.unsqueeze(1).bool()
    return (rank_i < rank_j) & mask_i & mask_j   # [B, K, K]


def _bpr_loss(scores: torch.Tensor, ranks: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    valid = _valid_pairs(ranks, cand_mask)   # [B, K, K]
    diff = (scores.unsqueeze(2) - scores.unsqueeze(1)).float()
    per_pair = -F.logsigmoid(diff)
    masked = torch.where(valid, per_pair, torch.zeros_like(per_pair))
    denom = valid.float().sum().clamp(min=1.0)
    return masked.sum() / denom


def _hinge_loss(
    scores: torch.Tensor,
    ranks:  torch.Tensor,
    cand_mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    valid = _valid_pairs(ranks, cand_mask)
    diff = (scores.unsqueeze(2) - scores.unsqueeze(1)).float()
    per_pair = F.relu(margin - diff)
    masked = torch.where(valid, per_pair, torch.zeros_like(per_pair))
    denom = valid.float().sum().clamp(min=1.0)
    return masked.sum() / denom


def compute_loss(
    scores:    torch.Tensor,   # [B, K]  model output
    batch:     Batch,
    loss_type: str = "bpr",
    margin:    float = 1.0,
) -> torch.Tensor:
    """
    Entry point called by train.py:

        loss = compute_loss(scores, batch, config.train.loss_type)
    """
    ranks      = batch.ranks           # [B, K]
    cand_mask  = batch.candidate_mask  # [B, K]  1=real, 0=pad

    if loss_type == "bpr":
        return _bpr_loss(scores, ranks, cand_mask)
    elif loss_type == "hinge":
        return _hinge_loss(scores, ranks, cand_mask, margin=margin)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Use 'bpr' or 'hinge'.")
