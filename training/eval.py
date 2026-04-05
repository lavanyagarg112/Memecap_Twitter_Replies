"""
  python eval.py --checkpoint checkpoints/best.pt --split test
  python eval.py --checkpoint checkpoints/best.pt --split val
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import load_datasets, make_collate_fn, MemeDataset
from dataset import _load_tasks
from model import build_model
from train import evaluate, _resolve_device, _format_metrics
from utils import load_checkpoint
from text_utils import Vocab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split",  default="test", choices=["val", "test"])
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = _resolve_device(args.device)

    raw   = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg   = Config.from_dict(raw["config"])
    vocab = Vocab.from_state_dict(raw["vocab"])

    model = build_model(cfg, vocab_size=len(vocab)).to(device)
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
        max_candidates = cfg.data.max_candidates_per_task,
    )
    collate = make_collate_fn(
        pipeline        = cfg.model.pipeline,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        vocab           = vocab,
        text_cfg        = cfg.text,
    )
    loader = DataLoader(
        ds,
        batch_size  = cfg.train.batch_size,
        shuffle     = False,
        num_workers = cfg.train.num_workers,
        collate_fn  = collate,
    )

    metrics = evaluate(model, loader, device, cfg)
    print(_format_metrics(f"{args.split.upper()} [{args.checkpoint}]", metrics))


if __name__ == "__main__":
    main()
