#!/usr/bin/env python3
"""
Create train/val/test CSV from meme_rankings.csv + MemeCap metadata.

Pointwise format: one row per tweet-meme pair with all features.
Split is done at the task level (all 10 candidates for a tweet stay together).

Usage:
    python create_train_data.py

Output:
    train.csv, val.csv, test.csv
"""

import csv
import json
import random
import requests
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# =========================
# CONFIG
# =========================

RANKINGS_CSV = "meme_rankings.csv"
ANNOTATIONS_CSV = "annotations_augmented.csv"
MEMECAP_URL = "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-trainval.json"

# Split ratios (by task, not by row)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42

OUTPUT_FIELDS = [
    "task_id",
    "tweet_text",
    "meme_post_id",
    "image_url",
    "meme_title",
    "img_captions",
    "meme_captions",
    "metaphors",
    "selection_method",
    "candidate_index",
    "rank",
    "avg_score",
    "num_votes",
    "num_yes",
    "num_no",
]


# =========================
# HELPERS
# =========================

def fetch_memecap() -> Dict[str, dict]:
    """Download MemeCap and build post_id -> metadata lookup."""
    print("Downloading MemeCap metadata...")
    r = requests.get(MEMECAP_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    lookup = {}
    for entry in data:
        pid = entry.get("post_id", "")
        if pid:
            lookup[pid] = entry
    print(f"  {len(lookup)} memes in lookup")
    return lookup


def format_captions(captions: Any) -> str:
    """Join list of captions into a single string."""
    if not captions:
        return ""
    if isinstance(captions, list):
        return " | ".join(str(c).strip() for c in captions if str(c).strip())
    return str(captions).strip()


def format_metaphors(metaphors: Any) -> str:
    """Format metaphor list into readable string."""
    if not metaphors or not isinstance(metaphors, list):
        return ""
    parts = []
    for m in metaphors:
        if isinstance(m, dict):
            met = m.get("metaphor", "")
            meaning = m.get("meaning", "")
            if met and meaning:
                parts.append(f"{met} -> {meaning}")
        else:
            parts.append(str(m))
    return " | ".join(parts)


# =========================
# MAIN
# =========================

def main():
    if not Path(RANKINGS_CSV).exists():
        print(f"Error: {RANKINGS_CSV} not found. Run the annotation pipeline first.")
        return

    # 1. Load MemeCap
    memecap = fetch_memecap()

    # 2. Find user-flagged meme_post_ids from annotations
    flagged_memes = set()
    if Path(ANNOTATIONS_CSV).exists():
        with open(ANNOTATIONS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("flag_inappropriate") == "1":
                    flagged_memes.add(row["meme_post_id"])
    print(f"  {len(flagged_memes)} memes flagged as inappropriate by annotators")

    # 3. Load rankings and group by task
    print(f"Loading {RANKINGS_CSV}...")
    task_rows: Dict[str, List[dict]] = defaultdict(list)

    with open(RANKINGS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["num_votes"]) == 0:
                continue
            task_rows[row["task_id"]].append(row)

    print(f"  {sum(len(v) for v in task_rows.values())} items across {len(task_rows)} tasks")

    # 4. Enrich with MemeCap metadata
    enriched_rows: Dict[str, List[dict]] = defaultdict(list)

    missing_meta = 0
    for task_id, rows in task_rows.items():
        for row in rows:
            meta = memecap.get(row["meme_post_id"], {})
            if not meta:
                missing_meta += 1

            enriched = {
                "task_id": row["task_id"],
                "tweet_text": row["tweet_text"],
                "meme_post_id": row["meme_post_id"],
                "image_url": row["image_url"],
                "meme_title": row["meme_title"],
                "img_captions": format_captions(meta.get("img_captions")),
                "meme_captions": format_captions(meta.get("meme_captions")),
                "metaphors": format_metaphors(meta.get("metaphors")),
                "selection_method": row["selection_method"],
                "candidate_index": row["candidate_index"],
                "rank": row["rank"],
                "avg_score": row["avg_score"],
                "num_votes": row["num_votes"],
                "num_yes": row["num_yes"],
                "num_no": row["num_no"],
            }
            enriched_rows[task_id].append(enriched)

    if missing_meta:
        print(f"  Warning: {missing_meta} memes not found in MemeCap (metadata will be empty)")

    # 5. Split by task
    task_ids = sorted(enriched_rows.keys())
    random.seed(SEED)
    random.shuffle(task_ids)

    n = len(task_ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": task_ids[:train_end],
        "val": task_ids[train_end:val_end],
        "test": task_ids[val_end:],
    }

    # 6. Write both versions: full and filtered (no flagged memes)
    for version, filter_flagged in [("", False), ("_clean", True)]:
        label = "clean (no flagged)" if filter_flagged else "full"
        print(f"\n  --- {label} ---")

        for split_name, split_tasks in splits.items():
            output_file = f"{split_name}{version}.csv"
            rows = []
            for task_id in split_tasks:
                for row in enriched_rows[task_id]:
                    if filter_flagged and row["meme_post_id"] in flagged_memes:
                        continue
                    rows.append(row)

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            n_tasks = len(set(r["task_id"] for r in rows))
            n_rows = len(rows)
            n_yes = sum(1 for r in rows if float(r["avg_score"]) > 0.5)
            print(f"  {output_file:18s}: {n_tasks:4d} tasks, {n_rows:5d} items, "
                  f"{n_yes} positive (avg_score > 0.5)")

    print(f"\nDone!")
    print(f"  Full:  train.csv, val.csv, test.csv")
    print(f"  Clean: train_clean.csv, val_clean.csv, test_clean.csv")


if __name__ == "__main__":
    main()
