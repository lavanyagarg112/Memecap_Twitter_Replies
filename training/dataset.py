from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset

from config import Config
from text_utils import Vocab, build_candidate_text, encode_text, normalize_text


@dataclass
class CandidateItem:
    candidate_id: str
    text: str
    rank: float
    avg_score: float
    metadata: dict[str, Any]
    input_ids: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)


@dataclass
class TaskItem:
    task_id: str
    context_text: str
    candidate_texts: list[str]
    ranks: list[float]
    avg_scores: list[float]
    candidate_ids: list[str]
    metadata: list[dict]
    context_input_ids: list[int] = field(default_factory=list)
    context_attention_mask: list[int] = field(default_factory=list)
    candidate_input_ids: list[list[int]] = field(default_factory=list)
    candidate_attention_mask: list[list[int]] = field(default_factory=list)


@dataclass
class Batch:
    context_input_ids: torch.Tensor
    context_attention_mask: torch.Tensor
    candidate_input_ids: torch.Tensor
    candidate_attention_mask: torch.Tensor
    candidate_mask: torch.Tensor
    ranks: torch.Tensor
    avg_scores: torch.Tensor
    task_ids: list[str]
    candidate_ids: list[list[str]]
    metadata: list[list[dict]]

    context_texts: list[str]
    image_urls: list[list[str]]


def load_csv_rows(csv_path: str) -> list[dict]:
    resolved_path = resolve_csv_path(csv_path)
    with open(resolved_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def resolve_csv_path(csv_path: str) -> str:
    if os.path.exists(csv_path):
        return csv_path

    candidates: list[str] = []
    normalized_path = csv_path.replace("\\", "/")
    if "/clean/" in normalized_path:
        candidates.append(normalized_path.replace("/clean/", "/", 1))
    else:
        directory, filename = os.path.split(normalized_path)
        candidates.append(os.path.join(directory, "clean", filename) if directory else os.path.join("data", "clean", filename))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find CSV file at '{csv_path}' or fallback locations: {candidates}")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def group_rows_by_task(rows: list[dict], min_candidates: int = 2) -> list[TaskItem]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        task_id = str(row.get("task_id", "") or "").strip()
        if not task_id:
            continue
        grouped.setdefault(task_id, []).append(row)

    tasks: list[TaskItem] = []
    for task_id, task_rows in grouped.items():
        if len(task_rows) < min_candidates:
            continue

        task_rows = sorted(
            task_rows,
            key=lambda row: (
                _to_float(row.get("rank"), float("inf")),
                _to_float(row.get("candidate_index"), float("inf")),
            ),
        )
        context_text = normalize_text(str(task_rows[0].get("tweet_text", "") or ""))

        candidate_texts: list[str] = []
        ranks: list[float] = []
        avg_scores: list[float] = []
        candidate_ids: list[str] = []
        metadata: list[dict] = []

        for row in task_rows:
            candidate_texts.append("")
            ranks.append(_to_float(row.get("rank"), default=9999.0))
            avg_scores.append(_to_float(row.get("avg_score"), default=0.0))
            candidate_ids.append(str(row.get("meme_post_id", "") or ""))
            metadata.append(dict(row))

        tasks.append(
            TaskItem(
                task_id=task_id,
                context_text=context_text,
                candidate_texts=candidate_texts,
                ranks=ranks,
                avg_scores=avg_scores,
                candidate_ids=candidate_ids,
                metadata=metadata,
            )
        )
    return tasks


def build_vocab_from_tasks(tasks: list[TaskItem], text_config) -> Vocab:
    vocab = Vocab(pad_token=text_config.pad_token, unk_token=text_config.unk_token)
    texts: list[str] = []
    for task in tasks:
        texts.append(task.context_text)
        texts.extend(task.candidate_texts)
    vocab.fit(texts, max_size=text_config.max_vocab_size)
    return vocab


class MemeRankingDataset(Dataset):
    def __init__(self, tasks: list[TaskItem], vocab: Vocab, text_config) -> None:
        self.tasks = tasks
        self.vocab = vocab
        self.text_config = text_config
        self._prepare()

    def _prepare(self) -> None:
        for task in self.tasks:
            context_ids, context_mask = encode_text(
                normalize_text(task.context_text, lowercase=self.text_config.lowercase),
                self.vocab,
                self.text_config.max_context_len,
            )
            task.context_input_ids = context_ids
            task.context_attention_mask = context_mask
            task.candidate_input_ids = []
            task.candidate_attention_mask = []

            for idx, candidate_text in enumerate(task.candidate_texts):
                text = normalize_text(candidate_text, lowercase=self.text_config.lowercase)
                input_ids, attention_mask = encode_text(
                    text,
                    self.vocab,
                    self.text_config.max_candidate_len,
                )
                task.candidate_input_ids.append(input_ids)
                task.candidate_attention_mask.append(attention_mask)
                task.candidate_texts[idx] = text

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> TaskItem:
        return self.tasks[index]


def collate_fn(batch: list[TaskItem]) -> Batch:
    batch_size = len(batch)
    max_candidates = max(len(task.candidate_input_ids) for task in batch)
    max_context_len = max(max(len(task.context_input_ids), 1) for task in batch)
    max_candidate_len = max(
        max(max((len(ids) for ids in task.candidate_input_ids), default=0), 1) for task in batch
    )

    context_input_ids = torch.zeros(batch_size, max_context_len, dtype=torch.long)
    context_attention_mask = torch.zeros(batch_size, max_context_len, dtype=torch.long)
    candidate_input_ids = torch.zeros(batch_size, max_candidates, max_candidate_len, dtype=torch.long)
    candidate_attention_mask = torch.zeros(batch_size, max_candidates, max_candidate_len, dtype=torch.long)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
    ranks = torch.full((batch_size, max_candidates), fill_value=9999.0, dtype=torch.float32)
    avg_scores = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
    task_ids: list[str] = []
    candidate_ids: list[list[str]] = []
    metadata: list[list[dict]] = []

    for batch_idx, task in enumerate(batch):
        context_len = len(task.context_input_ids)
        if context_len > 0:
            context_input_ids[batch_idx, :context_len] = torch.tensor(task.context_input_ids, dtype=torch.long)
            context_attention_mask[batch_idx, :context_len] = torch.tensor(task.context_attention_mask, dtype=torch.long)

        task_ids.append(task.task_id)
        candidate_ids.append(task.candidate_ids)
        metadata.append(task.metadata)

        for cand_idx, candidate_ids_list in enumerate(task.candidate_input_ids):
            cand_len = len(candidate_ids_list)
            if cand_len > 0:
                candidate_input_ids[batch_idx, cand_idx, :cand_len] = torch.tensor(candidate_ids_list, dtype=torch.long)
                candidate_attention_mask[batch_idx, cand_idx, :cand_len] = torch.tensor(
                    task.candidate_attention_mask[cand_idx],
                    dtype=torch.long,
                )
            candidate_mask[batch_idx, cand_idx] = 1.0
            ranks[batch_idx, cand_idx] = float(task.ranks[cand_idx])
            avg_scores[batch_idx, cand_idx] = float(task.avg_scores[cand_idx])

    return Batch(
        context_input_ids=context_input_ids,
        context_attention_mask=context_attention_mask,
        candidate_input_ids=candidate_input_ids,
        candidate_attention_mask=candidate_attention_mask,
        candidate_mask=candidate_mask,
        ranks=ranks,
        avg_scores=avg_scores,
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        metadata=metadata,
        
        context_texts=[task.context_text for task in batch],
        image_urls=[[meta.get("image_url", "") for meta in task.metadata] for task in batch],
    )


def load_datasets(config: Config) -> tuple[MemeRankingDataset, MemeRankingDataset, MemeRankingDataset, Vocab]:
    train_rows = load_csv_rows(config.data.train_csv)
    val_rows = load_csv_rows(config.data.val_csv)
    test_rows = load_csv_rows(config.data.test_csv)

    train_tasks = group_rows_by_task(train_rows, min_candidates=config.data.min_candidates_per_task)
    val_tasks = group_rows_by_task(val_rows, min_candidates=config.data.min_candidates_per_task)
    test_tasks = group_rows_by_task(test_rows, min_candidates=config.data.min_candidates_per_task)

    for task in train_tasks:
        task.candidate_texts = [
            normalize_text(build_candidate_text(meta, config.data.candidate_text_fields), lowercase=config.text.lowercase)
            for meta in task.metadata
        ]
    for task in val_tasks:
        task.candidate_texts = [
            normalize_text(build_candidate_text(meta, config.data.candidate_text_fields), lowercase=config.text.lowercase)
            for meta in task.metadata
        ]
    for task in test_tasks:
        task.candidate_texts = [
            normalize_text(build_candidate_text(meta, config.data.candidate_text_fields), lowercase=config.text.lowercase)
            for meta in task.metadata
        ]

    vocab = build_vocab_from_tasks(train_tasks, config.text)

    train_dataset = MemeRankingDataset(train_tasks, vocab, config.text)
    val_dataset = MemeRankingDataset(val_tasks, vocab, config.text)
    test_dataset = MemeRankingDataset(test_tasks, vocab, config.text)
    return train_dataset, val_dataset, test_dataset, vocab
