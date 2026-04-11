"""
  python eval.py --checkpoint checkpoints/best.pt --split test
  python eval.py --checkpoint checkpoints/best.pt --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import load_datasets, make_collate_fn, MemeDataset
from dataset import _load_tasks
from model import build_model
from train import evaluate, _resolve_device, _format_metrics
from utils import load_checkpoint
from text_utils import Vocab

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
            print(f"[eval] Remapped missing {kind} path '{saved_path}' -> '{candidate}'")
            return str(candidate)

    return saved_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split",  default="test", choices=["val", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--val_csv", default="")
    p.add_argument("--test_csv", default="")
    p.add_argument("--image_dir", default="")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--qwen_pair_chunk_size", type=int, default=0)
    args = p.parse_args()

    device = _resolve_device(args.device)

    raw   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg   = Config.from_dict(raw["config"])
    vocab = Vocab.from_state_dict(raw["vocab"])

    cfg.data.val_csv = args.val_csv or _resolve_path(cfg.data.val_csv, kind="val_csv")
    cfg.data.test_csv = args.test_csv or _resolve_path(cfg.data.test_csv, kind="test_csv")
    cfg.data.image_dir = args.image_dir or _resolve_path(cfg.data.image_dir, kind="image_dir")

    model = build_model(cfg, vocab_size=len(vocab)).to(device)
    if args.qwen_pair_chunk_size > 0:
        model.qwen_pair_chunk_size = args.qwen_pair_chunk_size
    load_checkpoint(args.checkpoint, model, map_location=device)

    from dataset import _build_processors
    tokenizer, image_processor = _build_processors(cfg)

    csv_path = cfg.data.val_csv if args.split == "val" else cfg.data.test_csv
    tasks = _load_tasks(csv_path, cfg.data.candidate_text_fields)
    ds    = MemeDataset(
        tasks          = tasks,
        pipeline       = cfg.model.pipeline,
        image_dir      = cfg.data.image_dir,
        min_candidates = cfg.data.min_candidates_per_task,
    )
    collate = make_collate_fn(
        pipeline        = cfg.model.pipeline,
        encoder_type    = cfg.model.encoder_type,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        vocab           = vocab,
        text_cfg        = cfg.text,
    )
    loader = DataLoader(
        ds,
        batch_size  = args.batch_size or cfg.train.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate,
    )

    metrics = evaluate(model, loader, device, cfg)
    print(_format_metrics(f"{args.split.upper()} [{args.checkpoint}]", metrics))


if __name__ == "__main__":
    main()
