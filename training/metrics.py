from __future__ import annotations

import math

import torch

from dataset import Batch


def _get_valid_task_tensors(
    scores: torch.Tensor,
    values: torch.Tensor,
    candidate_mask: torch.Tensor,
):
    for task_scores, task_values, task_mask in zip(scores, values, candidate_mask):
        valid = task_mask > 0
        yield task_scores[valid], task_values[valid]


def score_at_1(scores: torch.Tensor, avg_scores: torch.Tensor, candidate_mask: torch.Tensor) -> float:
    total = 0.0
    count = 0
    for task_scores, task_avg_scores in _get_valid_task_tensors(scores, avg_scores, candidate_mask):
        if task_scores.numel() == 0:
            continue
        chosen = int(torch.argmax(task_scores).item())
        total += float(task_avg_scores[chosen].item())
        count += 1
    return total / max(count, 1)


def recall_at_1(
    scores: torch.Tensor,
    ranks: torch.Tensor,
    avg_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    target: str = "rank1",
) -> float:
    hits = 0.0
    count = 0

    for task_scores, task_ranks, task_avg_scores, task_mask in zip(scores, ranks, avg_scores, candidate_mask):
        valid = task_mask > 0
        task_scores = task_scores[valid]
        task_ranks = task_ranks[valid]
        task_avg_scores = task_avg_scores[valid]
        if task_scores.numel() == 0:
            continue
        pred_idx = int(torch.argmax(task_scores).item())
        if target == "rank1":
            best_mask = task_ranks == task_ranks.min()
        elif target == "avg_score":
            best_mask = task_avg_scores == task_avg_scores.max()
        else:
            raise ValueError(f"Unsupported recall target: {target}")
        hits += float(best_mask[pred_idx].item())
        count += 1
    return hits / max(count, 1)


def mean_reciprocal_rank(
    scores: torch.Tensor,
    ranks: torch.Tensor,
    avg_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    target: str = "rank1",
) -> float:
    reciprocal_sum = 0.0
    count = 0

    for task_scores, task_ranks, task_avg_scores, task_mask in zip(scores, ranks, avg_scores, candidate_mask):
        valid = task_mask > 0
        task_scores = task_scores[valid]
        task_ranks = task_ranks[valid]
        task_avg_scores = task_avg_scores[valid]
        if task_scores.numel() == 0:
            continue

        order = torch.argsort(task_scores, descending=True)
        if target == "rank1":
            target_mask = task_ranks == task_ranks.min()
        elif target == "avg_score":
            target_mask = task_avg_scores == task_avg_scores.max()
        else:
            raise ValueError(f"Unsupported MRR target: {target}")

        reciprocal_rank = 0.0
        for rank_idx, item_idx in enumerate(order.tolist(), start=1):
            if bool(target_mask[item_idx].item()):
                reciprocal_rank = 1.0 / rank_idx
                break
        reciprocal_sum += reciprocal_rank
        count += 1
    return reciprocal_sum / max(count, 1)


def ndcg_at_k(scores: torch.Tensor, avg_scores: torch.Tensor, candidate_mask: torch.Tensor, k: int = 5) -> float:
    total = 0.0
    count = 0

    for task_scores, task_avg_scores in _get_valid_task_tensors(scores, avg_scores, candidate_mask):
        if task_scores.numel() == 0:
            continue

        pred_order = torch.argsort(task_scores, descending=True)
        true_order = torch.argsort(task_avg_scores, descending=True)
        top_k = min(k, task_scores.numel())

        dcg = 0.0
        idcg = 0.0
        for idx in range(top_k):
            pred_rel = float(task_avg_scores[pred_order[idx]].item())
            ideal_rel = float(task_avg_scores[true_order[idx]].item())
            denom = math.log2(idx + 2)
            dcg += (2.0 ** pred_rel - 1.0) / denom
            idcg += (2.0 ** ideal_rel - 1.0) / denom

        total += 0.0 if idcg <= 0.0 else dcg / idcg
        count += 1
    return total / max(count, 1)


def consensus_hit_rate(
    scores: torch.Tensor,
    avg_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    threshold: float = 0.8,
) -> float:
    hits = 0.0
    count = 0
    for task_scores, task_avg_scores in _get_valid_task_tensors(scores, avg_scores, candidate_mask):
        if task_scores.numel() == 0:
            continue
        pred_idx = int(torch.argmax(task_scores).item())
        hits += float(task_avg_scores[pred_idx].item() >= threshold)
        count += 1
    return hits / max(count, 1)


def compute_metrics(scores: torch.Tensor, batch: Batch, eval_config) -> dict[str, float]:
    return {
        "score_at_1": score_at_1(scores, batch.avg_scores, batch.candidate_mask),
        "recall_at_1": recall_at_1(
            scores,
            batch.ranks,
            batch.avg_scores,
            batch.candidate_mask,
            target=eval_config.recall_target,
        ),
        "mrr": mean_reciprocal_rank(
            scores,
            batch.ranks,
            batch.avg_scores,
            batch.candidate_mask,
            target=eval_config.recall_target,
        ),
        "ndcg_at_k": ndcg_at_k(scores, batch.avg_scores, batch.candidate_mask, k=eval_config.ndcg_k),
        "consensus_hit_rate": consensus_hit_rate(
            scores,
            batch.avg_scores,
            batch.candidate_mask,
            threshold=eval_config.consensus_threshold,
        ),
    }
