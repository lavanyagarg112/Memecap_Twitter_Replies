from __future__ import annotations

import argparse
from dataclasses import fields

import torch

from config import Config, parse_args
from dataset import MemeRankingDataset, TaskItem, collate_fn
from model import build_model
from text_utils import Vocab, build_candidate_text, normalize_text
from train import _resolve_device
from utils import load_checkpoint, move_batch_to_device


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


def rank_candidates(model, vocab: Vocab, config: Config, tweet_text: str, candidate_rows: list[dict]) -> list[dict]:
    candidate_texts = [
        normalize_text(build_candidate_text(row, config.data.candidate_text_fields), lowercase=config.text.lowercase)
        for row in candidate_rows
    ]
    task = TaskItem(
        task_id="inference_task",
        context_text=normalize_text(tweet_text, lowercase=config.text.lowercase),
        candidate_texts=candidate_texts,
        ranks=[9999.0] * len(candidate_rows),
        avg_scores=[0.0] * len(candidate_rows),
        candidate_ids=[str(row.get("meme_post_id", idx)) for idx, row in enumerate(candidate_rows)],
        metadata=[dict(row) for row in candidate_rows],
    )
    dataset = MemeRankingDataset([task], vocab, config.text)
    batch = collate_fn([dataset[0]])

    device = next(model.parameters()).device
    batch = move_batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        scores = model(batch)[0].detach().cpu()

    ranked: list[dict] = []
    for idx, row in enumerate(candidate_rows):
        item = dict(row)
        item["candidate_text"] = candidate_texts[idx]
        item["score"] = float(scores[idx].item())
        ranked.append(item)
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for meme reply selection.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tweet_text", type=str, required=True)
    parser.add_argument("--candidates_json", type=str, required=True, help="JSON file containing a list of candidate meme rows.")
    parser.add_argument("--top_k", type=int, default=5)
    args, remaining = parser.parse_known_args()

    import json
    import sys

    original_argv = sys.argv
    sys.argv = [original_argv[0]] + remaining
    config = parse_args()
    sys.argv = original_argv

    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = _apply_checkpoint_config(config, checkpoint_meta.get("config", {}))
    vocab = Vocab.from_state_dict(checkpoint_meta["vocab"])

    device = _resolve_device(config.train.device)
    model = build_model(config, vocab_size=len(vocab)).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    with open(args.candidates_json, "r", encoding="utf-8") as handle:
        candidate_rows = json.load(handle)
    if not isinstance(candidate_rows, list):
        raise ValueError("candidates_json must contain a JSON list of candidate rows.")

    ranked = rank_candidates(model, vocab, config, args.tweet_text, candidate_rows)
    for idx, item in enumerate(ranked[: args.top_k], start=1):
        meme_id = item.get("meme_post_id", f"cand_{idx}")
        title = item.get("meme_title", "")
        print(f"{idx}. meme_post_id={meme_id} score={item['score']:.4f} title={title}")


if __name__ == "__main__":
    main()
