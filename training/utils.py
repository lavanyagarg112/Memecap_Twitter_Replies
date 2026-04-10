from __future__ import annotations

import os
import random
import tempfile
import warnings
from dataclasses import asdict, is_dataclass
from typing import Optional

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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch_state = state["torch"]
        if not isinstance(torch_state, torch.Tensor):
            try:
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            except Exception:
                warnings.warn("Skipping invalid CPU RNG state in checkpoint resume.")
                torch_state = None
        elif torch_state.dtype != torch.uint8:
            torch_state = torch_state.to(dtype=torch.uint8)
        if torch_state is not None:
            try:
                torch.set_rng_state(torch_state.cpu())
            except Exception:
                warnings.warn("Skipping malformed CPU RNG state in checkpoint resume.")
    if torch.cuda.is_available() and "cuda" in state:
        cuda_state = state["cuda"]
        if isinstance(cuda_state, (list, tuple)):
            normalized = []
            for item in cuda_state:
                if not isinstance(item, torch.Tensor):
                    try:
                        item = torch.tensor(item, dtype=torch.uint8)
                    except Exception:
                        warnings.warn("Skipping invalid CUDA RNG state in checkpoint resume.")
                        normalized = []
                        break
                elif item.dtype != torch.uint8:
                    item = item.to(dtype=torch.uint8)
                normalized.append(item.cpu())
            if normalized:
                try:
                    torch.cuda.set_rng_state_all(normalized)
                except Exception:
                    warnings.warn("Skipping malformed CUDA RNG state in checkpoint resume.")
        else:
            warnings.warn("Skipping malformed CUDA RNG state in checkpoint resume.")


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    def _m(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return t.to(device) if t is not None else None

    batch.context_input_ids        = _m(batch.context_input_ids)
    batch.context_attention_mask   = _m(batch.context_attention_mask)
    batch.candidate_input_ids      = _m(batch.candidate_input_ids)
    batch.candidate_attention_mask = _m(batch.candidate_attention_mask)
    batch.pixel_values             = _m(batch.pixel_values)
    batch.image_grid_thw           = _m(batch.image_grid_thw)
    batch.candidate_mask           = batch.candidate_mask.to(device)
    batch.ranks                    = batch.ranks.to(device)
    return batch


def save_checkpoint(
    path:         str,
    model:        torch.nn.Module,
    optimizer,
    scheduler,
    epoch:        int,
    config:       Config,
    vocab:        Vocab,
    best_metric:  Optional[float] = None,
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch":                epoch,
        "best_metric":          best_metric,
        "rng_state":            capture_rng_state(),
        "config":               asdict(config) if is_dataclass(config) else config,
        "vocab":                vocab.state_dict(),
    }
    dirpath = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_ckpt_", suffix=".pt", dir=dirpath)
    os.close(fd)
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_checkpoint(
    path:         str,
    model:        torch.nn.Module,
    optimizer=None,
    scheduler=None,
    map_location=None,
) -> dict:
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
        self.sum   = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum   += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
