from __future__ import annotations

import math
from typing import Dict

import torch

from dataset import Batch


def _top1_idx(scores: torch.Tensor, mask: torch.Tensor) -> int:

    masked = scores.clone()
    masked[mask == 0] = float("-inf")
    return int(masked.argmax().item())


def _recall_at_1(scores: torch.Tensor, ranks: torch.Tensor, mask: torch.Tensor) -> float:
    idx = _top1_idx(scores, mask)
    return float(int(ranks[idx].item()) == 1)


def _mrr(scores: torch.Tensor, ranks: torch.Tensor, mask: torch.Tensor) -> float:
    masked = scores.clone()
    masked[mask == 0] = float("-inf")
    order = masked.argsort(descending=True)
    for pos, idx in enumerate(order):
        if mask[idx] == 0:
            continue
        if int(ranks[idx].item()) == 1:
            return 1.0 / (pos + 1)
    return 0.0


def _ndcg_at_k(
    scores: torch.Tensor,
    ranks:  torch.Tensor,
    mask:   torch.Tensor,
    k:      int = 10,
) -> float:
    real_idx   = (mask == 1).nonzero(as_tuple=True)[0]
    if len(real_idx) == 0:
        return 0.0

    real_scores = scores[real_idx]
    real_ranks  = ranks[real_idx]
    K = min(k, len(real_idx))

    pred_order  = real_scores.argsort(descending=True)[:K]
    ideal_order = real_ranks.argsort()[:K] # ascending rank = best first

    def dcg(indices):
        total = 0.0
        for pos, i in enumerate(indices):
            rel = 1.0 / float(real_ranks[i].item())
            total += (2 ** rel - 1) / math.log2(pos + 2)
        return total

    idcg = dcg(ideal_order)
    return dcg(pred_order) / idcg if idcg > 0 else 0.0


def _score_at_1(scores: torch.Tensor, ranks: torch.Tensor, mask: torch.Tensor) -> float:
    idx = _top1_idx(scores, mask)
    return 1.0 / float(ranks[idx].item())


def compute_metrics(
    scores:   torch.Tensor,   # [B, K]
    batch:    Batch,
    eval_cfg,
) -> Dict[str, float]:
    B         = scores.shape[0]
    ranks     = batch.ranks           # [B, K]
    cand_mask = batch.candidate_mask  # [B, K]
    k         = eval_cfg.ndcg_k

    r1_sum = mrr_sum = ndcg_sum = s1_sum = 0.0

    for b in range(B):
        s = scores[b]
        r = ranks[b]
        m = cand_mask[b]

        r1_sum   += _recall_at_1(s, r, m)
        mrr_sum  += _mrr(s, r, m)
        ndcg_sum += _ndcg_at_k(s, r, m, k=k)
        s1_sum   += _score_at_1(s, r, m)

    return {
        "recall_at_1":    r1_sum   / B,
        "mrr":            mrr_sum  / B,
        f"ndcg_{k}":      ndcg_sum / B,
        "score_at_1":     s1_sum   / B,
    }
