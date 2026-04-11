from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import MemeDataset, _build_processors, _load_tasks, make_collate_fn
from losses import compute_loss
from metrics import compute_metrics
from model import build_model
from text_utils import Vocab
from train import _amp_context, _resolve_amp_settings, _resolve_device
from utils import AverageMeter, load_checkpoint, move_batch_to_device


_TRAINING_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TRAINING_DIR.parent


def _resolve_path(saved_path: str, *, kind: str) -> str:
    if not saved_path:
        return saved_path

    path = Path(saved_path)
    candidates: list[Path] = []

    if path.is_absolute():
        if path.exists():
            return str(path)
        parts = path.parts
        if "training" in parts:
            idx = parts.index("training")
            candidates.append(_PROJECT_ROOT.joinpath(*parts[idx:]))
        if "data" in parts:
            idx = parts.index("data")
            candidates.append(_TRAINING_DIR.joinpath(*parts[idx:]))
    else:
        candidates.append(_PROJECT_ROOT / path)
        candidates.append(_TRAINING_DIR / path)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            print(f"[analysis] Remapped missing {kind} path '{saved_path}' -> '{candidate}'")
            return str(candidate)

    return saved_path


def _parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid --run value '{spec}'. Expected format label=checkpoint_path."
        )
    label, checkpoint = spec.split("=", 1)
    label = label.strip()
    checkpoint = checkpoint.strip()
    if not label or not checkpoint:
        raise argparse.ArgumentTypeError(
            f"Invalid --run value '{spec}'. Expected format label=checkpoint_path."
        )
    return label, checkpoint


def _load_task_rows(csv_path: str, candidate_text_fields: list[str]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_id = row["task_id"]
            if task_id not in grouped:
                grouped[task_id] = {
                    "task_id": task_id,
                    "tweet_text": row["tweet_text"],
                    "candidates": [],
                }

            parts = [row.get(field, "").strip() for field in candidate_text_fields]
            candidate_text = " | ".join(part for part in parts if part)
            grouped[task_id]["candidates"].append(
                {
                    "meme_post_id": row["meme_post_id"],
                    "image_url": row.get("image_url", ""),
                    "candidate_text": candidate_text,
                    "meme_title": row.get("meme_title", "").strip(),
                    "img_captions": row.get("img_captions", "").strip(),
                    "meme_captions": row.get("meme_captions", "").strip(),
                    "metaphors": row.get("metaphors", "").strip(),
                    "selection_method": row.get("selection_method", "").strip(),
                    "candidate_index": row.get("candidate_index", "").strip(),
                    "rank": int(row["rank"]),
                    "similarity_score": row.get("similarity_score", "").strip(),
                }
            )

    for task in grouped.values():
        task["candidates"].sort(key=lambda candidate: candidate["rank"])

    return grouped


def _rank_bucket(rank: int) -> str:
    if rank <= 1:
        return "1"
    if rank == 2:
        return "2"
    if rank == 3:
        return "3"
    return "4+"


def _outcome_bucket(rank: int) -> str:
    if rank <= 1:
        return "correct"
    if rank <= 3:
        return "near_miss"
    return "bad_miss"


def _tweet_length_bucket(length: int) -> str:
    if length <= 80:
        return "0-80"
    if length <= 120:
        return "81-120"
    if length <= 160:
        return "121-160"
    return "161+"


def _candidate_text_bucket(length: float) -> str:
    if length <= 200:
        return "0-200"
    if length <= 275:
        return "201-275"
    if length <= 350:
        return "276-350"
    return "351+"


def _metaphor_coverage_bucket(coverage: float) -> str:
    if coverage <= 0:
        return "none"
    if coverage >= 0.999:
        return "all"
    return "partial"


def _safe_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _median(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.median(items) if items else 0.0


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_fields(top_k: int) -> list[str]:
    fields: list[str] = []
    for idx in range(1, top_k + 1):
        fields.extend(
            [
                f"pred_{idx}_meme_post_id",
                f"pred_{idx}_title",
                f"pred_{idx}_score",
                f"pred_{idx}_gold_rank",
            ]
        )
    return fields


def _task_summary_fieldnames(top_k: int) -> list[str]:
    return [
        "model_label",
        "split",
        "task_id",
        "tweet_text",
        "tweet_length_chars",
        "has_hashtag",
        "num_candidates",
        "avg_candidate_text_chars",
        "max_candidate_text_chars",
        "metaphor_coverage",
        "metaphor_coverage_bucket",
        "gold_top_meme_post_id",
        "gold_top_title",
        "gold_top_predicted_position",
        "top1_gold_rank",
        "top1_gold_rank_bucket",
        "outcome_bucket",
        "top1_top2_margin",
        "all_predicted_ids_json",
        "all_predicted_scores_json",
        "all_predicted_gold_ranks_json",
        "all_gold_ranks_json",
    ] + _prediction_fields(top_k)


def _candidate_fieldnames() -> list[str]:
    return [
        "model_label",
        "split",
        "task_id",
        "tweet_text",
        "meme_post_id",
        "meme_title",
        "candidate_text",
        "candidate_text_chars",
        "metaphors",
        "has_metaphor",
        "selection_method",
        "candidate_index",
        "similarity_score",
        "score",
        "gold_rank",
        "predicted_position",
        "is_predicted_top1",
        "is_gold_top1",
    ]


def _load_run(
    *,
    label: str,
    checkpoint: str,
    split: str,
    device: torch.device,
    batch_size_override: int,
    num_workers: int,
    qwen_pair_chunk_size: int,
    val_csv_override: str,
    test_csv_override: str,
    image_dir_override: str,
    top_k: int,
) -> tuple[list[dict], list[dict], dict]:
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config.from_dict(raw["config"])
    vocab = Vocab.from_state_dict(raw["vocab"])

    cfg.data.val_csv = val_csv_override or _resolve_path(cfg.data.val_csv, kind="val_csv")
    cfg.data.test_csv = test_csv_override or _resolve_path(cfg.data.test_csv, kind="test_csv")
    cfg.data.image_dir = image_dir_override or _resolve_path(cfg.data.image_dir, kind="image_dir")

    csv_path = cfg.data.val_csv if split == "val" else cfg.data.test_csv
    raw_task_map = _load_task_rows(csv_path, cfg.data.candidate_text_fields)

    model = build_model(cfg, vocab_size=len(vocab)).to(device)
    if qwen_pair_chunk_size > 0:
        model.qwen_pair_chunk_size = qwen_pair_chunk_size
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    tokenizer, image_processor = _build_processors(cfg)
    tasks = _load_tasks(csv_path, cfg.data.candidate_text_fields)
    dataset = MemeDataset(
        tasks=tasks,
        pipeline=cfg.model.pipeline,
        image_dir=cfg.data.image_dir,
        min_candidates=cfg.data.min_candidates_per_task,
    )
    collate = make_collate_fn(
        pipeline=cfg.model.pipeline,
        encoder_type=cfg.model.encoder_type,
        tokenizer=tokenizer,
        image_processor=image_processor,
        vocab=vocab,
        text_cfg=cfg.text,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size_override or cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    task_rows: list[dict] = []
    candidate_rows: list[dict] = []
    loss_meter = AverageMeter()
    metric_meters: dict[str, AverageMeter] = {}
    use_amp, amp_dtype, _ = _resolve_amp_settings(cfg, device)

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with _amp_context(use_amp, amp_dtype):
                scores = model(batch)
                loss = compute_loss(
                    scores,
                    batch,
                    cfg.train.loss_type,
                    margin=cfg.train.hinge_margin,
                )

            batch_size = len(batch.task_ids)
            loss_meter.update(float(loss.item()), batch_size)
            metrics = compute_metrics(scores, batch, cfg.eval)
            for name, value in metrics.items():
                metric_meters.setdefault(name, AverageMeter()).update(value, batch_size)

            scores_cpu = scores.detach().float().cpu()
            ranks_cpu = batch.ranks.detach().cpu()
            masks_cpu = batch.candidate_mask.detach().cpu()

            for batch_idx, task_id in enumerate(batch.task_ids):
                task_meta = raw_task_map[task_id]
                valid_count = int(masks_cpu[batch_idx].sum().item())
                valid_scores = scores_cpu[batch_idx, :valid_count].tolist()
                valid_ranks = [int(value) for value in ranks_cpu[batch_idx, :valid_count].tolist()]
                candidates = task_meta["candidates"][:valid_count]

                predicted_order = sorted(
                    range(valid_count),
                    key=lambda idx: valid_scores[idx],
                    reverse=True,
                )
                predicted_position = {idx: pos + 1 for pos, idx in enumerate(predicted_order)}
                gold_top_idx = min(range(valid_count), key=lambda idx: valid_ranks[idx])

                candidate_text_lengths = [len(candidate["candidate_text"]) for candidate in candidates]
                metaphor_flags = [bool(candidate["metaphors"]) for candidate in candidates]
                metaphor_coverage = (
                    sum(1 for flag in metaphor_flags if flag) / valid_count if valid_count else 0.0
                )
                top1_idx = predicted_order[0]
                top1_gold_rank = valid_ranks[top1_idx]
                margin = (
                    valid_scores[predicted_order[0]] - valid_scores[predicted_order[1]]
                    if len(predicted_order) > 1
                    else 0.0
                )

                task_row = {
                    "model_label": label,
                    "split": split,
                    "task_id": task_id,
                    "tweet_text": task_meta["tweet_text"],
                    "tweet_length_chars": len(task_meta["tweet_text"].strip()),
                    "has_hashtag": "#" in task_meta["tweet_text"],
                    "num_candidates": valid_count,
                    "avg_candidate_text_chars": round(_mean(candidate_text_lengths), 4),
                    "max_candidate_text_chars": max(candidate_text_lengths) if candidate_text_lengths else 0,
                    "metaphor_coverage": round(metaphor_coverage, 4),
                    "metaphor_coverage_bucket": _metaphor_coverage_bucket(metaphor_coverage),
                    "gold_top_meme_post_id": candidates[gold_top_idx]["meme_post_id"],
                    "gold_top_title": candidates[gold_top_idx]["meme_title"],
                    "gold_top_predicted_position": predicted_position[gold_top_idx],
                    "top1_gold_rank": top1_gold_rank,
                    "top1_gold_rank_bucket": _rank_bucket(top1_gold_rank),
                    "outcome_bucket": _outcome_bucket(top1_gold_rank),
                    "top1_top2_margin": round(margin, 6),
                    "all_predicted_ids_json": json.dumps(
                        [candidates[idx]["meme_post_id"] for idx in predicted_order],
                        ensure_ascii=False,
                    ),
                    "all_predicted_scores_json": json.dumps(
                        [round(valid_scores[idx], 6) for idx in predicted_order]
                    ),
                    "all_predicted_gold_ranks_json": json.dumps(
                        [valid_ranks[idx] for idx in predicted_order]
                    ),
                    "all_gold_ranks_json": json.dumps(valid_ranks),
                }

                for pred_idx in range(top_k):
                    field_prefix = f"pred_{pred_idx + 1}"
                    if pred_idx < len(predicted_order):
                        candidate_idx = predicted_order[pred_idx]
                        task_row[f"{field_prefix}_meme_post_id"] = candidates[candidate_idx]["meme_post_id"]
                        task_row[f"{field_prefix}_title"] = candidates[candidate_idx]["meme_title"]
                        task_row[f"{field_prefix}_score"] = round(valid_scores[candidate_idx], 6)
                        task_row[f"{field_prefix}_gold_rank"] = valid_ranks[candidate_idx]
                    else:
                        task_row[f"{field_prefix}_meme_post_id"] = ""
                        task_row[f"{field_prefix}_title"] = ""
                        task_row[f"{field_prefix}_score"] = ""
                        task_row[f"{field_prefix}_gold_rank"] = ""

                task_rows.append(task_row)

                for idx, candidate in enumerate(candidates):
                    candidate_rows.append(
                        {
                            "model_label": label,
                            "split": split,
                            "task_id": task_id,
                            "tweet_text": task_meta["tweet_text"],
                            "meme_post_id": candidate["meme_post_id"],
                            "meme_title": candidate["meme_title"],
                            "candidate_text": candidate["candidate_text"],
                            "candidate_text_chars": len(candidate["candidate_text"]),
                            "metaphors": candidate["metaphors"],
                            "has_metaphor": bool(candidate["metaphors"]),
                            "selection_method": candidate["selection_method"],
                            "candidate_index": candidate["candidate_index"],
                            "similarity_score": candidate["similarity_score"],
                            "score": round(valid_scores[idx], 6),
                            "gold_rank": valid_ranks[idx],
                            "predicted_position": predicted_position[idx],
                            "is_predicted_top1": predicted_position[idx] == 1,
                            "is_gold_top1": valid_ranks[idx] == 1,
                        }
                    )

    metrics_summary = {"loss": round(loss_meter.avg, 6)}
    for name, meter in metric_meters.items():
        metrics_summary[name] = round(meter.avg, 6)

    del loader
    del dataset
    del tokenizer
    del image_processor
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return task_rows, candidate_rows, metrics_summary


def _build_histogram_rows(task_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["model_label"], row["split"])].append(row)

    histogram_rows: list[dict] = []
    for (label, split), rows in sorted(grouped.items()):
        counter = Counter(row["top1_gold_rank_bucket"] for row in rows)
        total = len(rows)
        for bucket in ["1", "2", "3", "4+"]:
            count = counter.get(bucket, 0)
            histogram_rows.append(
                {
                    "model_label": label,
                    "split": split,
                    "top1_gold_rank_bucket": bucket,
                    "count": count,
                    "rate": round(count / total, 6) if total else 0.0,
                }
            )
    return histogram_rows


def _build_confidence_rows(task_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["model_label"], row["split"], row["outcome_bucket"])].append(row)

    confidence_rows: list[dict] = []
    for (label, split, bucket), rows in sorted(grouped.items()):
        margins = [float(row["top1_top2_margin"]) for row in rows]
        top1_ranks = [int(row["top1_gold_rank"]) for row in rows]
        confidence_rows.append(
            {
                "model_label": label,
                "split": split,
                "outcome_bucket": bucket,
                "count": len(rows),
                "mean_margin": round(_mean(margins), 6),
                "median_margin": round(_median(margins), 6),
                "max_margin": round(max(margins), 6) if margins else 0.0,
                "mean_top1_gold_rank": round(_mean(top1_ranks), 6),
            }
        )
    return confidence_rows


def _build_slice_rows(task_rows: list[dict]) -> list[dict]:
    bucketed_rows: list[tuple[str, str, str, str, dict]] = []
    for row in task_rows:
        bucketed_rows.extend(
            [
                (
                    row["model_label"],
                    row["split"],
                    "num_candidates",
                    str(row["num_candidates"]),
                    row,
                ),
                (
                    row["model_label"],
                    row["split"],
                    "tweet_length_bucket",
                    _tweet_length_bucket(int(row["tweet_length_chars"])),
                    row,
                ),
                (
                    row["model_label"],
                    row["split"],
                    "avg_candidate_text_bucket",
                    _candidate_text_bucket(float(row["avg_candidate_text_chars"])),
                    row,
                ),
                (
                    row["model_label"],
                    row["split"],
                    "metaphor_coverage_bucket",
                    row["metaphor_coverage_bucket"],
                    row,
                ),
                (
                    row["model_label"],
                    row["split"],
                    "has_hashtag",
                    "yes" if row["has_hashtag"] else "no",
                    row,
                ),
            ]
        )

    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for label, split, slice_name, slice_value, row in bucketed_rows:
        grouped[(label, split, slice_name, slice_value)].append(row)

    slice_rows: list[dict] = []
    for (label, split, slice_name, slice_value), rows in sorted(grouped.items()):
        top1_ranks = [int(row["top1_gold_rank"]) for row in rows]
        margins = [float(row["top1_top2_margin"]) for row in rows]
        slice_rows.append(
            {
                "model_label": label,
                "split": split,
                "slice_name": slice_name,
                "slice_value": slice_value,
                "count": len(rows),
                "recall_at_1": round(
                    sum(1 for rank in top1_ranks if rank == 1) / len(rows),
                    6,
                ),
                "mean_top1_gold_rank": round(_mean(top1_ranks), 6),
                "mean_margin": round(_mean(margins), 6),
            }
        )
    return slice_rows


def _build_cross_model_rows(task_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    labels = sorted({row["model_label"] for row in task_rows})
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in task_rows:
        grouped[(row["split"], row["task_id"])][row["model_label"]] = row

    task_agreement_rows: list[dict] = []
    summary_counters: dict[str, Counter] = defaultdict(Counter)

    for (split, task_id), rows_by_label in sorted(grouped.items()):
        base_row = next(iter(rows_by_label.values()))
        agreement_row = {
            "split": split,
            "task_id": task_id,
            "tweet_text": base_row["tweet_text"],
            "num_models_present": len(rows_by_label),
            "models_present_json": json.dumps(sorted(rows_by_label), ensure_ascii=False),
        }

        present_top_ids = []
        top1_ranks = {}
        for label in labels:
            row = rows_by_label.get(label)
            if row is None:
                agreement_row[f"{label}_top1_meme_post_id"] = ""
                agreement_row[f"{label}_top1_gold_rank"] = ""
                agreement_row[f"{label}_outcome_bucket"] = ""
                agreement_row[f"{label}_margin"] = ""
                agreement_row[f"{label}_correct"] = ""
                continue

            present_top_ids.append(row["pred_1_meme_post_id"])
            top1_ranks[label] = int(row["top1_gold_rank"])
            agreement_row[f"{label}_top1_meme_post_id"] = row["pred_1_meme_post_id"]
            agreement_row[f"{label}_top1_gold_rank"] = row["top1_gold_rank"]
            agreement_row[f"{label}_outcome_bucket"] = row["outcome_bucket"]
            agreement_row[f"{label}_margin"] = row["top1_top2_margin"]
            agreement_row[f"{label}_correct"] = row["top1_gold_rank"] == 1

        agreement_row["all_models_present"] = len(rows_by_label) == len(labels)
        agreement_row["correct_models_count"] = sum(
            1 for rank in top1_ranks.values() if rank == 1
        )
        agreement_row["all_correct"] = (
            len(rows_by_label) == len(labels)
            and all(rank == 1 for rank in top1_ranks.values())
        )
        agreement_row["all_wrong"] = (
            len(rows_by_label) == len(labels)
            and all(rank != 1 for rank in top1_ranks.values())
        )
        agreement_row["all_same_top1_prediction"] = (
            len(set(present_top_ids)) == 1 if present_top_ids else False
        )
        agreement_row["best_models_json"] = json.dumps(
            sorted(
                [
                    label
                    for label, rank in top1_ranks.items()
                    if rank == min(top1_ranks.values())
                ]
            ),
            ensure_ascii=False,
        ) if top1_ranks else "[]"
        task_agreement_rows.append(agreement_row)

        if agreement_row["all_models_present"]:
            counter = summary_counters[split]
            counter["tasks_with_all_models"] += 1
            counter["all_correct"] += int(agreement_row["all_correct"])
            counter["all_wrong"] += int(agreement_row["all_wrong"])
            counter["all_same_top1_prediction"] += int(agreement_row["all_same_top1_prediction"])

            if {"text", "image"}.issubset(top1_ranks):
                counter["image_better_than_text"] += int(top1_ranks["image"] < top1_ranks["text"])
                counter["text_better_than_image"] += int(top1_ranks["text"] < top1_ranks["image"])
                counter["text_image_tie"] += int(top1_ranks["text"] == top1_ranks["image"])
            if {"text", "image", "multimodal"}.issubset(top1_ranks):
                counter["multimodal_better_than_both"] += int(
                    top1_ranks["multimodal"] < min(top1_ranks["text"], top1_ranks["image"])
                )
                counter["multimodal_worse_than_both"] += int(
                    top1_ranks["multimodal"] > max(top1_ranks["text"], top1_ranks["image"])
                )
                counter["text_only_correct"] += int(
                    top1_ranks["text"] == 1
                    and top1_ranks["image"] != 1
                    and top1_ranks["multimodal"] != 1
                )
                counter["image_only_correct"] += int(
                    top1_ranks["image"] == 1
                    and top1_ranks["text"] != 1
                    and top1_ranks["multimodal"] != 1
                )
                counter["multimodal_only_correct"] += int(
                    top1_ranks["multimodal"] == 1
                    and top1_ranks["text"] != 1
                    and top1_ranks["image"] != 1
                )

    summary_rows: list[dict] = []
    for split, counter in sorted(summary_counters.items()):
        summary_rows.append(
            {
                "split": split,
                "tasks_with_all_models": counter["tasks_with_all_models"],
                "all_correct": counter["all_correct"],
                "all_wrong": counter["all_wrong"],
                "all_same_top1_prediction": counter["all_same_top1_prediction"],
                "text_only_correct": counter["text_only_correct"],
                "image_only_correct": counter["image_only_correct"],
                "multimodal_only_correct": counter["multimodal_only_correct"],
                "image_better_than_text": counter["image_better_than_text"],
                "text_better_than_image": counter["text_better_than_image"],
                "text_image_tie": counter["text_image_tie"],
                "multimodal_better_than_both": counter["multimodal_better_than_both"],
                "multimodal_worse_than_both": counter["multimodal_worse_than_both"],
            }
        )

    return task_agreement_rows, summary_rows


def _build_audit_rows(task_rows: list[dict], audit_examples: int) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["model_label"], row["split"])].append(row)

    audit_rows: list[dict] = []
    for (label, split), rows in sorted(grouped.items()):
        bucket_rows = {
            "success": [row for row in rows if int(row["top1_gold_rank"]) == 1],
            "near_miss": [row for row in rows if int(row["top1_gold_rank"]) in {2, 3}],
            "confident_failure": [row for row in rows if int(row["top1_gold_rank"]) >= 4],
        }
        for audit_bucket, candidates in bucket_rows.items():
            selected = sorted(
                candidates,
                key=lambda row: float(row["top1_top2_margin"]),
                reverse=True,
            )[:audit_examples]
            for row in selected:
                audit_row = {
                    "model_label": label,
                    "split": split,
                    "audit_bucket": audit_bucket,
                    "task_id": row["task_id"],
                    "tweet_text": row["tweet_text"],
                    "top1_gold_rank": row["top1_gold_rank"],
                    "top1_top2_margin": row["top1_top2_margin"],
                    "gold_top_meme_post_id": row["gold_top_meme_post_id"],
                    "gold_top_title": row["gold_top_title"],
                    "gold_top_predicted_position": row["gold_top_predicted_position"],
                    "suggested_reason": "",
                }
                for idx in range(1, 4):
                    audit_row[f"pred_{idx}_meme_post_id"] = row.get(f"pred_{idx}_meme_post_id", "")
                    audit_row[f"pred_{idx}_title"] = row.get(f"pred_{idx}_title", "")
                    audit_row[f"pred_{idx}_score"] = row.get(f"pred_{idx}_score", "")
                    audit_row[f"pred_{idx}_gold_rank"] = row.get(f"pred_{idx}_gold_rank", "")
                audit_rows.append(audit_row)
    return audit_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump per-task prediction artifacts and build error-analysis summaries."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT",
        help="Model label and checkpoint path. Repeat for multiple models.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["val", "test"],
        help="Dataset split to analyze. Repeat for multiple splits. Default: val and test.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_csv", default="")
    parser.add_argument("--test_csv", default="")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--qwen_pair_chunk_size", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--audit_examples", type=int, default=5)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    run_specs = [_parse_run_spec(spec) for spec in args.run]
    splits = args.split or ["val", "test"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_rows: list[dict] = []
    candidate_rows: list[dict] = []
    metrics_rows: list[dict] = []

    for label, checkpoint in run_specs:
        for split in splits:
            print(f"[analysis] Running label={label} split={split} checkpoint={checkpoint}")
            run_task_rows, run_candidate_rows, metrics_summary = _load_run(
                label=label,
                checkpoint=checkpoint,
                split=split,
                device=device,
                batch_size_override=args.batch_size,
                num_workers=args.num_workers,
                qwen_pair_chunk_size=args.qwen_pair_chunk_size,
                val_csv_override=args.val_csv,
                test_csv_override=args.test_csv,
                image_dir_override=args.image_dir,
                top_k=args.top_k,
            )
            task_rows.extend(run_task_rows)
            candidate_rows.extend(run_candidate_rows)
            metrics_rows.append(
                {
                    "model_label": label,
                    "split": split,
                    **metrics_summary,
                }
            )

    histogram_rows = _build_histogram_rows(task_rows)
    confidence_rows = _build_confidence_rows(task_rows)
    slice_rows = _build_slice_rows(task_rows)
    cross_task_rows, cross_summary_rows = _build_cross_model_rows(task_rows)
    audit_rows = _build_audit_rows(task_rows, args.audit_examples)

    _write_csv(
        output_dir / "metrics_summary.csv",
        metrics_rows,
        fieldnames=["model_label", "split", "loss", "recall_at_1", "mrr", "ndcg_10", "score_at_1"],
    )
    _write_csv(
        output_dir / "task_predictions.csv",
        task_rows,
        fieldnames=_task_summary_fieldnames(args.top_k),
    )
    _write_csv(
        output_dir / "candidate_scores.csv",
        candidate_rows,
        fieldnames=_candidate_fieldnames(),
    )
    _write_csv(
        output_dir / "top1_gold_rank_histogram.csv",
        histogram_rows,
        fieldnames=["model_label", "split", "top1_gold_rank_bucket", "count", "rate"],
    )
    _write_csv(
        output_dir / "confidence_summary.csv",
        confidence_rows,
        fieldnames=[
            "model_label",
            "split",
            "outcome_bucket",
            "count",
            "mean_margin",
            "median_margin",
            "max_margin",
            "mean_top1_gold_rank",
        ],
    )
    _write_csv(
        output_dir / "slice_analysis.csv",
        slice_rows,
        fieldnames=[
            "model_label",
            "split",
            "slice_name",
            "slice_value",
            "count",
            "recall_at_1",
            "mean_top1_gold_rank",
            "mean_margin",
        ],
    )
    _write_csv(
        output_dir / "cross_model_task_agreement.csv",
        cross_task_rows,
        fieldnames=list(cross_task_rows[0].keys()) if cross_task_rows else ["split", "task_id"],
    )
    _write_csv(
        output_dir / "cross_model_summary.csv",
        cross_summary_rows,
        fieldnames=list(cross_summary_rows[0].keys()) if cross_summary_rows else ["split"],
    )
    _write_csv(
        output_dir / "qualitative_audit.csv",
        audit_rows,
        fieldnames=list(audit_rows[0].keys()) if audit_rows else ["model_label", "split", "audit_bucket"],
    )

    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "runs": [{"label": label, "checkpoint": checkpoint} for label, checkpoint in run_specs],
                "splits": splits,
                "device": str(device),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "qwen_pair_chunk_size": args.qwen_pair_chunk_size,
                "top_k": args.top_k,
                "audit_examples": args.audit_examples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[analysis] Wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
