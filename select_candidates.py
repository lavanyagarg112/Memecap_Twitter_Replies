"""
Select 9 distractor memes for each tweet-meme pair.

Strategy: mixed (4 semantically similar + 5 random).
Uses sentence-transformers for text-based semantic similarity
across all memes in the MemeCap dataset.

Install:
    pip install requests numpy sentence-transformers

Run:
    python select_candidates.py
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================

NUM_SEMANTIC = 4
NUM_RANDOM = 5
TOTAL_CANDIDATES = 1 + NUM_SEMANTIC + NUM_RANDOM  # 10

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DATASET_SPLIT = "trainval"
DATASET_URLS = {
    "test": "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-test.json",
    "trainval": "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-trainval.json",
}

TWEET_FILES = [
    "train_memecap_tweets_clean.jsonl",
    "train_memecap_tweets_indirect_clean.jsonl",
]

OUTPUT_FILE = "annotation_tasks.json"
SEED = 42


# =========================
# HELPERS
# =========================

def fetch_memecap() -> List[Dict[str, Any]]:
    """Download MemeCap dataset from GitHub."""
    url = DATASET_URLS[DATASET_SPLIT]
    print(f"Downloading MemeCap ({DATASET_SPLIT}) from {url} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    print(f"  Loaded {len(data)} memes.")
    return data


def load_tweets() -> List[Dict[str, Any]]:
    """Load tweet records from JSONL files, skipping errors."""
    tweets = []
    for fname in TWEET_FILES:
        path = Path(fname)
        if not path.exists():
            print(f"  Warning: {fname} not found, skipping.")
            continue
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "error" in rec or "tweet_that_meme_replies_to" not in rec:
                    continue
                tweets.append({
                    "post_id": rec["post_id"],
                    "tweet_text": rec["tweet_that_meme_replies_to"],
                    "title": rec.get("title", ""),
                    "image_url": rec.get("image_url", ""),
                })
                count += 1
        print(f"  Loaded {count} tweets from {fname}")
    return tweets


def to_str(val: Any) -> str:
    """Safely convert any value to string, handling NaN/floats/None."""
    if val is None:
        return ""
    if isinstance(val, float):
        return "" if np.isnan(val) else str(val)
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def build_meme_text(meme: Dict[str, Any]) -> str:
    """Build a text representation of a meme for embedding."""
    parts = []

    title = to_str(meme.get("title", ""))
    if title:
        parts.append(title)

    for cap in meme.get("img_captions", []):
        s = to_str(cap)
        if s:
            parts.append(s)

    for m in meme.get("metaphors", []):
        if isinstance(m, dict):
            met = to_str(m.get("metaphor", ""))
            meaning = to_str(m.get("meaning", ""))
            if met and meaning:
                parts.append(f"{met} means {meaning}")

    return " ".join(parts) if parts else "unknown meme"


def compute_embeddings(model: SentenceTransformer, memes: List[Dict]) -> np.ndarray:
    """Compute normalized embeddings for all memes."""
    texts = [build_meme_text(m) for m in memes]
    print(f"  Encoding {len(texts)} meme texts ...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)


def select_distractors(
    target_idx: int,
    similarity_matrix: np.ndarray,
    num_semantic: int,
    num_random: int,
    rng: random.Random,
) -> tuple:
    """
    Select distractor indices using mixed strategy.
    Returns (semantic_indices, random_indices).
    """
    n = similarity_matrix.shape[0]
    sims = similarity_matrix[target_idx]

    # Top-k most similar (excluding self)
    sorted_indices = np.argsort(-sims)
    semantic = []
    for idx in sorted_indices:
        if idx == target_idx:
            continue
        semantic.append(int(idx))
        if len(semantic) >= num_semantic:
            break

    # Random from the rest
    excluded = set(semantic) | {target_idx}
    available = [i for i in range(n) if i not in excluded]
    rand_picks = rng.sample(available, min(num_random, len(available)))

    return semantic, rand_picks


# =========================
# MAIN
# =========================

def main():
    rng = random.Random(SEED)

    # 1. Load data
    print("[Step 1/5] Loading data ...")
    memes = fetch_memecap()
    tweets = load_tweets()
    print(f"  Total: {len(memes)} memes, {len(tweets)} tweets")

    if not tweets:
        print("No tweets loaded. Make sure the JSONL files exist.")
        return

    # 2. Build post_id -> meme index mapping
    print(f"\n[Step 2/5] Building post_id -> meme index mapping ...")
    post_id_to_idx = {}
    for i, m in enumerate(memes):
        pid = m.get("post_id", "")
        if pid:
            post_id_to_idx[pid] = i
    print(f"  Mapped {len(post_id_to_idx)} unique post_ids.")

    # 3. Compute embeddings & similarity matrix
    print(f"\n[Step 3/5] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = compute_embeddings(model, memes)

    print("  Computing similarity matrix ...")
    sim_matrix = embeddings @ embeddings.T
    print(f"  Similarity matrix shape: {sim_matrix.shape}")

    # 4. Build annotation tasks
    print(f"\n[Step 4/5] Selecting candidates for {len(tweets)} tweets ...")
    tasks = []
    skipped = 0

    for i, tweet in enumerate(tweets):
        post_id = tweet["post_id"]
        if post_id not in post_id_to_idx:
            skipped += 1
            print(f"  [{i+1}/{len(tweets)}] SKIP post_id={post_id} — not found in MemeCap")
            continue

        target_idx = post_id_to_idx[post_id]
        semantic_idxs, random_idxs = select_distractors(
            target_idx, sim_matrix, NUM_SEMANTIC, NUM_RANDOM, rng
        )

        # Build candidate list
        candidates = []

        # Original (correct) meme
        orig = memes[target_idx]
        candidates.append({
            "meme_post_id": orig["post_id"],
            "image_url": orig.get("url", ""),
            "title": orig.get("title", ""),
            "selection_method": "original",
        })

        # Semantic distractors
        semantic_titles = []
        for idx in semantic_idxs:
            m = memes[idx]
            candidates.append({
                "meme_post_id": m["post_id"],
                "image_url": m.get("url", ""),
                "title": m.get("title", ""),
                "selection_method": "semantic",
            })
            semantic_titles.append(m.get("title", "?"))

        # Random distractors
        for idx in random_idxs:
            m = memes[idx]
            candidates.append({
                "meme_post_id": m["post_id"],
                "image_url": m.get("url", ""),
                "title": m.get("title", ""),
                "selection_method": "random",
            })

        # Shuffle so annotators can't guess position
        rng.shuffle(candidates)

        tasks.append({
            "task_id": post_id,
            "tweet_text": tweet["tweet_text"],
            "post_id": post_id,
            "candidates": candidates,
        })

        # Log progress
        tweet_preview = tweet["tweet_text"][:60] + ("..." if len(tweet["tweet_text"]) > 60 else "")
        print(f"  [{i+1}/{len(tweets)}] post_id={post_id}")
        print(f"    Tweet: {tweet_preview}")
        print(f"    Original meme: \"{orig.get('title', '?')}\"")
        print(f"    Semantic distractors: {semantic_titles}")

        if (i + 1) % 100 == 0:
            print(f"  --- Progress: {i+1}/{len(tweets)} tweets processed, {skipped} skipped ---")

    # 5. Save
    print(f"\n[Step 5/5] Saving to {OUTPUT_FILE} ...")
    output = {
        "tasks": tasks,
        "metadata": {
            "total_tasks": len(tasks),
            "candidates_per_task": TOTAL_CANDIDATES,
            "num_semantic_distractors": NUM_SEMANTIC,
            "num_random_distractors": NUM_RANDOM,
            "embedding_model": EMBEDDING_MODEL,
            "seed": SEED,
            "tweet_files": TWEET_FILES,
            "skipped_tweets": skipped,
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Tasks created: {len(tasks)}")
    print(f"  Skipped (post_id not in MemeCap): {skipped}")
    print(f"  Candidates per task: {TOTAL_CANDIDATES} (1 original + {NUM_SEMANTIC} semantic + {NUM_RANDOM} random)")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
