from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass

import torch

from config import Config
from dataset import Batch
from text_utils import Vocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    batch.context_input_ids = batch.context_input_ids.to(device)
    batch.context_attention_mask = batch.context_attention_mask.to(device)
    batch.candidate_input_ids = batch.candidate_input_ids.to(device)
    batch.candidate_attention_mask = batch.candidate_attention_mask.to(device)
    batch.candidate_mask = batch.candidate_mask.to(device)
    batch.ranks = batch.ranks.to(device)
    batch.avg_scores = batch.avg_scores.to(device)
    return batch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, config: Config, vocab: Vocab, best_metric: float | None = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": asdict(config) if is_dataclass(config) else config,
        "vocab": vocab.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location=None) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count
