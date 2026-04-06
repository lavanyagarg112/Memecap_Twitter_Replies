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
    score_i = scores.unsqueeze(2)   # [B, K, 1]
    score_j = scores.unsqueeze(1)   # [B, 1, K]
    valid   = _valid_pairs(ranks, cand_mask).float()
    per_pair = -F.logsigmoid(score_i - score_j)
    denom    = valid.sum().clamp(min=1.0)
    return (per_pair * valid).sum() / denom


def _hinge_loss(
    scores: torch.Tensor,
    ranks:  torch.Tensor,
    cand_mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    score_i = scores.unsqueeze(2)
    score_j = scores.unsqueeze(1)
    valid   = _valid_pairs(ranks, cand_mask).float()
    per_pair = F.relu(margin - (score_i - score_j))
    denom    = valid.sum().clamp(min=1.0)
    return (per_pair * valid).sum() / denom


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
