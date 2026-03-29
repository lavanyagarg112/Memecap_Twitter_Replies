from __future__ import annotations

import argparse
from dataclasses import fields

import torch
from torch.utils.data import DataLoader

from config import Config, parse_args
from dataset import collate_fn, load_datasets
from train import _resolve_device, evaluate
from model import build_model
from utils import load_checkpoint


def _apply_checkpoint_config(config: Config, checkpoint_config: dict) -> Config:
    if not checkpoint_config:
        return config

    for section_name in ("data", "text", "model", "train", "eval"):
        section = getattr(config, section_name)
        section_values = checkpoint_config.get(section_name, {})
        valid_names = {field.name for field in fields(section)}
        for key, value in section_values.items():
            if key in valid_names:
                setattr(section, key, value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained meme reply ranking model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args, remaining = parser.parse_known_args()

    import sys

    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining
    config = parse_args()
    sys.argv = original_argv

    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = _apply_checkpoint_config(config, checkpoint_meta.get("config", {}))

    train_dataset, val_dataset, test_dataset, vocab = load_datasets(config)
    target_dataset = val_dataset if args.split == "val" else test_dataset
    dataloader = DataLoader(
        target_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
    )

    device = _resolve_device(config.train.device)
    model = build_model(config, vocab_size=len(vocab)).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate(model, dataloader, device, config)

    print(f"Split={args.split}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
