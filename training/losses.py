from __future__ import annotations

import torch
import torch.nn.functional as F

from dataset import Batch


def pairwise_hinge_loss(
    scores: torch.Tensor,
    ranks: torch.Tensor,
    candidate_mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    total_loss = scores.new_tensor(0.0)
    total_pairs = 0

    for task_scores, task_ranks, task_mask in zip(scores, ranks, candidate_mask):
        valid = task_mask > 0
        valid_scores = task_scores[valid]
        valid_ranks = task_ranks[valid]
        if valid_scores.numel() < 2:
            continue

        preferred = valid_ranks.unsqueeze(1) < valid_ranks.unsqueeze(0)
        if not preferred.any():
            continue

        pairwise_margin = margin - (valid_scores.unsqueeze(1) - valid_scores.unsqueeze(0))
        loss_matrix = F.relu(pairwise_margin)
        total_loss = total_loss + loss_matrix[preferred].sum()
        total_pairs += int(preferred.sum().item())

    if total_pairs == 0:
        return scores.new_tensor(0.0)
    return total_loss / total_pairs


def bpr_loss(scores: torch.Tensor, ranks: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    total_loss = scores.new_tensor(0.0)
    total_pairs = 0

    for task_scores, task_ranks, task_mask in zip(scores, ranks, candidate_mask):
        valid = task_mask > 0
        valid_scores = task_scores[valid]
        valid_ranks = task_ranks[valid]
        if valid_scores.numel() < 2:
            continue

        preferred = valid_ranks.unsqueeze(1) < valid_ranks.unsqueeze(0)
        if not preferred.any():
            continue

        score_diff = valid_scores.unsqueeze(1) - valid_scores.unsqueeze(0)
        loss_matrix = -F.logsigmoid(score_diff)
        total_loss = total_loss + loss_matrix[preferred].sum()
        total_pairs += int(preferred.sum().item())

    if total_pairs == 0:
        return scores.new_tensor(0.0)
    return total_loss / total_pairs


def compute_loss(scores: torch.Tensor, batch: Batch, loss_type: str) -> torch.Tensor:
    if loss_type == "hinge":
        return pairwise_hinge_loss(scores=scores, ranks=batch.ranks, candidate_mask=batch.candidate_mask)
    if loss_type == "bpr":
        return bpr_loss(scores=scores, ranks=batch.ranks, candidate_mask=batch.candidate_mask)
    raise ValueError(f"Unsupported loss_type: {loss_type}")
