"""
  # Pipeline 1 — text only (HF sentence-transformer)
  python train.py --pipeline text --encoder_type hf

  # Pipeline 1 — lightweight BOW baseline
  python train.py --pipeline text --encoder_type bow_mean

  # Pipeline 1 — GRU baseline
  python train.py --pipeline text --encoder_type gru

  # Pipeline 2 — image only (Qwen2.5-VL cross-encoder)
  python train.py --pipeline image --encoder_type qwen_vl --image_dir data/images

  # Pipeline 3 — multimodal Qwen2.5-VL cross-encoder
  python train.py --pipeline multimodal --encoder_type qwen_vl --image_dir data/images

  # Override data paths
  python train.py --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv
"""

from __future__ import annotations

import math
import os

import torch
from torch.utils.data import DataLoader

from config import parse_args
from dataset import collate_fn, load_datasets
from losses import compute_loss
from metrics import compute_metrics
from model import build_model
from utils import (
    AverageMeter,
    ensure_dir,
    load_checkpoint,
    move_batch_to_device,
    save_checkpoint,
    set_seed,
)


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_scheduler(optimizer, config, steps_per_epoch: int):
    total_steps  = config.train.num_epochs * steps_per_epoch
    warmup_steps = min(config.train.warmup_steps, total_steps // 10)

    if warmup_steps > 0:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, config) -> dict:
    model.train()
    loss_meter    = AverageMeter()
    metric_meters: dict[str, AverageMeter] = {}

    use_amp = config.train.use_amp and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            scores = model(batch)
            loss   = compute_loss(scores, batch, config.train.loss_type)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = len(batch.task_ids)
        loss_meter.update(float(loss.item()), batch_size)

        with torch.no_grad():
            metrics = compute_metrics(scores.detach(), batch, config.eval)
        for name, value in metrics.items():
            metric_meters.setdefault(name, AverageMeter()).update(value, batch_size)

    results = {"loss": loss_meter.avg}
    for name, meter in metric_meters.items():
        results[name] = meter.avg
    return results


@torch.no_grad()
def evaluate(model, dataloader, device, config) -> dict:
    model.eval()
    loss_meter    = AverageMeter()
    metric_meters: dict[str, AverageMeter] = {}

    for batch in dataloader:
        batch  = move_batch_to_device(batch, device)
        scores = model(batch)
        loss   = compute_loss(scores, batch, config.train.loss_type)

        batch_size = len(batch.task_ids)
        loss_meter.update(float(loss.item()), batch_size)

        metrics = compute_metrics(scores, batch, config.eval)
        for name, value in metrics.items():
            metric_meters.setdefault(name, AverageMeter()).update(value, batch_size)

    results = {"loss": loss_meter.avg}
    for name, meter in metric_meters.items():
        results[name] = meter.avg
    return results


def _format_metrics(prefix: str, metrics: dict) -> str:
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    return f"{prefix}: " + " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    config = parse_args()
    set_seed(config.train.seed)
    ensure_dir(config.train.save_dir)
    device = _resolve_device(config.train.device)

    # ── data ──────────────────────────────────────────────────────────────────
    train_dataset, val_dataset, test_dataset, vocab = load_datasets(config)

    # collate_fn is set as a module-level variable by load_datasets
    from dataset import collate_fn as _collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.train.batch_size,
        shuffle     = True,
        num_workers = config.train.num_workers,
        collate_fn  = _collate_fn,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.train.batch_size,
        shuffle     = False,
        num_workers = config.train.num_workers,
        collate_fn  = _collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config.train.batch_size,
        shuffle     = False,
        num_workers = config.train.num_workers,
        collate_fn  = _collate_fn,
    )

    # ── model + optimiser ─────────────────────────────────────────────────────
    model     = build_model(config, vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.train.lr,
        weight_decay = config.train.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    # ── checkpoint paths ──────────────────────────────────────────────────────
    latest_path = os.path.join(config.train.save_dir, "latest.pt")
    best_path   = os.path.join(config.train.save_dir, "best.pt")
    # Primary metric: recall_at_1 (rank-based; no avg_score in the dataset)
    best_score  = float("-inf")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, config.train.num_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        val_metrics = evaluate(model, val_loader, device, config)

        print(_format_metrics(f"Epoch {epoch} Train", train_metrics))
        print(_format_metrics(f"Epoch {epoch} Val",   val_metrics))

        # Save latest every epoch
        save_checkpoint(
            latest_path,
            model        = model,
            optimizer    = optimizer,
            scheduler    = scheduler,
            epoch        = epoch,
            config       = config,
            vocab        = vocab,
            best_metric  = best_score,
        )

        # Save best when recall_at_1 improves
        current_score = val_metrics.get("recall_at_1", float("-inf"))
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(
                best_path,
                model        = model,
                optimizer    = optimizer,
                scheduler    = scheduler,
                epoch        = epoch,
                config       = config,
                vocab        = vocab,
                best_metric  = best_score,
            )
            print(f"  ★ New best  recall_at_1={best_score:.4f}  → {best_path}")

    print("\nLoading best checkpoint for test evaluation...")
    load_checkpoint(best_path, model, map_location=device)
    test_metrics = evaluate(model, test_loader, device, config)
    print(_format_metrics("Best Checkpoint Test", test_metrics))


if __name__ == "__main__":
    main()
