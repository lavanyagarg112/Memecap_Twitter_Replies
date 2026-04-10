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

import contextlib
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
    restore_rng_state,
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


def _resolve_resume_path(config, latest_path: str) -> str | None:
    resume = config.train.resume_from
    if not resume:
        return None
    return latest_path if resume == "auto" else resume


def _resume_training_state(model, optimizer, scheduler, device, config, latest_path: str, best_path: str):
    resume_path = _resolve_resume_path(config, latest_path)
    if resume_path is None:
        return 1, float("-inf"), None
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    try:
        checkpoint = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        restore_rng_state(checkpoint.get("rng_state"))
    except Exception as exc:
        can_fallback = (
            resume_path == latest_path
            and os.path.exists(best_path)
            and best_path != latest_path
        )
        if not can_fallback:
            raise
        print(
            f"Warning: failed to resume from {resume_path}: {exc}. "
            f"Falling back to {best_path}."
        )
        checkpoint = load_checkpoint(
            best_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        restore_rng_state(checkpoint.get("rng_state"))
        resume_path = best_path

    last_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = last_epoch + 1
    best_score = float(checkpoint.get("best_metric", float("-inf")))
    print(
        f"Resumed training from {resume_path} "
        f"(last_epoch={last_epoch}, next_epoch={start_epoch}, best_recall_at_1={best_score:.4f})"
    )
    return start_epoch, best_score, resume_path


def _resolve_amp_settings(config, device: torch.device) -> tuple[bool, torch.dtype | None, bool]:
    if device.type != "cuda" or not config.train.use_amp:
        return False, None, False

    amp_dtype = config.train.amp_dtype
    if amp_dtype == "auto":
        if config.model.encoder_type == "qwen_vl" and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return True, torch.float16, True
    if amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 autocast requested, but this CUDA device does not support bf16.")
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _amp_context(enabled: bool, amp_dtype: torch.dtype | None):
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=amp_dtype)


def _assert_finite(name: str, tensor: torch.Tensor, batch, *, step: int, epoch: int) -> None:
    if torch.isfinite(tensor).all():
        return

    detached = tensor.detach().float()
    finite = detached[torch.isfinite(detached)]
    stats = (
        f"finite_min={finite.min().item():.4f}, finite_max={finite.max().item():.4f}"
        if finite.numel() > 0
        else "no finite values"
    )
    sample_task_ids = ",".join(batch.task_ids[:3])
    raise FloatingPointError(
        f"Non-finite {name} detected at epoch={epoch} step={step} "
        f"pipeline_batch_tasks={sample_task_ids} shape={tuple(tensor.shape)} {stats}"
    )


def train_one_epoch(model, dataloader, optimizer, scheduler, device, config, epoch: int) -> dict:
    model.train()
    loss_meter    = AverageMeter()
    metric_meters: dict[str, AverageMeter] = {}

    use_amp, amp_dtype, use_scaler = _resolve_amp_settings(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None

    for step_idx, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with _amp_context(use_amp, amp_dtype):
            scores = model(batch)
            loss   = compute_loss(
                scores,
                batch,
                config.train.loss_type,
                margin=config.train.hinge_margin,
            )

        _assert_finite("scores", scores, batch, step=step_idx, epoch=epoch)
        _assert_finite("loss", loss, batch, step=step_idx, epoch=epoch)

        if scaler is not None and use_scaler:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            scale_before = None
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)

        if scaler is not None and use_scaler:
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
        else:
            optimizer.step()
            optimizer_stepped = True

        if optimizer_stepped:
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
def evaluate(model, dataloader, device, config, epoch: int | None = None) -> dict:
    model.eval()
    loss_meter    = AverageMeter()
    metric_meters: dict[str, AverageMeter] = {}
    use_amp, amp_dtype, _ = _resolve_amp_settings(config, device)

    for step_idx, batch in enumerate(dataloader, start=1):
        batch  = move_batch_to_device(batch, device)
        with _amp_context(use_amp, amp_dtype):
            scores = model(batch)
            loss   = compute_loss(
                scores,
                batch,
                config.train.loss_type,
                margin=config.train.hinge_margin,
            )

        _assert_finite("eval_scores", scores, batch, step=step_idx, epoch=epoch or 0)
        _assert_finite("eval_loss", loss, batch, step=step_idx, epoch=epoch or 0)

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
    start_epoch, best_score, resume_path = _resume_training_state(
        model, optimizer, scheduler, device, config, latest_path, best_path
    )

    # ── training loop ─────────────────────────────────────────────────────────
    if start_epoch > config.train.num_epochs:
        print(
            f"Resume checkpoint already reached epoch {start_epoch - 1}, "
            f"which is >= requested num_epochs={config.train.num_epochs}. Skipping training."
        )

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, config, epoch
        )
        val_metrics = evaluate(model, val_loader, device, config, epoch=epoch)

        print(_format_metrics(f"Epoch {epoch} Train", train_metrics))
        print(_format_metrics(f"Epoch {epoch} Val",   val_metrics))

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

        # Save latest every epoch after updating best_score
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

    eval_path = best_path if os.path.exists(best_path) else (resume_path or latest_path)
    print(f"\nLoading checkpoint for test evaluation: {eval_path}")
    load_checkpoint(eval_path, model, map_location=device)
    test_metrics = evaluate(model, test_loader, device, config)
    label = "Best Checkpoint Test" if eval_path == best_path else "Latest Checkpoint Test"
    print(_format_metrics(label, test_metrics))


if __name__ == "__main__":
    main()
