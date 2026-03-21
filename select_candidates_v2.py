"""
Select 9 distractor memes for each tweet-meme pair.
V2: Reads flagged_memes.json and excludes confirmed-offensive memes.

- Excluded memes are skipped as candidates AND as originals
  (if the original meme is excluded, the entire tweet is dropped)
- Outputs to annotation_tasks_clean.json

Install:
    pip install requests numpy sentence-transformers

Run:
    python flag_memes.py        # review flagged memes first
    python select_candidates_v2.py
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set

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

FLAGGED_FILE = "flagged_memes.json"
OUTPUT_FILE = "annotation_tasks_clean.json"
SEED = 42


# =========================
# HELPERS
# =========================

def fetch_memecap() -> List[Dict[str, Any]]:
    url = DATASET_URLS[DATASET_SPLIT]
    print(f"Downloading MemeCap ({DATASET_SPLIT}) from {url} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    print(f"  Loaded {len(data)} memes.")
    return data


def load_excluded() -> Set[str]:
    """Load excluded post_ids from flagged_memes.json."""
    path = Path(FLAGGED_FILE)
    if not path.exists():
        print(f"  Warning: {FLAGGED_FILE} not found. No memes will be excluded.")
        print(f"  Run 'python flag_memes.py' first to review flagged content.")
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    excluded = set(data.get("excluded_post_ids", []))
    summary = data.get("summary", {})
    print(f"  Loaded blocklist: {len(excluded)} excluded, "
          f"{summary.get('kept', '?')} kept, "
          f"{summary.get('pending_review', '?')} pending")
    return excluded


def load_tweets() -> List[Dict[str, Any]]:
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
    if val is None:
        return ""
    if isinstance(val, float):
        return "" if np.isnan(val) else str(val)
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def build_meme_text(meme: Dict[str, Any]) -> str:
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
    texts = [build_meme_text(m) for m in memes]
    print(f"  Encoding {len(texts)} meme texts ...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)


def select_distractors(
    target_idx: int,
    similarity_matrix: np.ndarray,
    num_semantic: int,
    num_random: int,
    excluded_idxs: Set[int],
    rng: random.Random,
) -> tuple:
    """Select distractors, skipping excluded memes."""
    sims = similarity_matrix[target_idx]
    sorted_indices = np.argsort(-sims)

    semantic = []
    for idx in sorted_indices:
        if idx == target_idx or idx in excluded_idxs:
            continue
        semantic.append(int(idx))
        if len(semantic) >= num_semantic:
            break

    skip = set(semantic) | {target_idx} | excluded_idxs
    available = [i for i in range(similarity_matrix.shape[0]) if i not in skip]
    rand_picks = rng.sample(available, min(num_random, len(available)))

    return semantic, rand_picks


# =========================
# MAIN
# =========================

def main():
    rng = random.Random(SEED)

    # 1. Load data
    print("[Step 1/6] Loading data ...")
    memes = fetch_memecap()
    excluded_ids = load_excluded()
    tweets = load_tweets()
    print(f"  Total: {len(memes)} memes, {len(tweets)} tweets, {len(excluded_ids)} excluded")

    if not tweets:
        print("No tweets loaded. Make sure the JSONL files exist.")
        return

    # 2. Build mappings
    print(f"\n[Step 2/6] Building post_id -> meme index mapping ...")
    post_id_to_idx = {}
    for i, m in enumerate(memes):
        pid = m.get("post_id", "")
        if pid:
            post_id_to_idx[pid] = i

    # Build set of excluded indices for fast lookup
    excluded_idxs = {post_id_to_idx[pid] for pid in excluded_ids if pid in post_id_to_idx}
    print(f"  Mapped {len(post_id_to_idx)} post_ids, {len(excluded_idxs)} excluded indices.")

    # 3. Compute embeddings & similarity matrix
    print(f"\n[Step 3/6] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = compute_embeddings(model, memes)

    print("  Computing similarity matrix ...")
    sim_matrix = embeddings @ embeddings.T
    print(f"  Similarity matrix shape: {sim_matrix.shape}")

    # 4. Build annotation tasks
    print(f"\n[Step 4/6] Selecting candidates for {len(tweets)} tweets ...")
    tasks = []
    skipped = 0
    skipped_excluded = 0

    for i, tweet in enumerate(tweets):
        post_id = tweet["post_id"]

        if post_id not in post_id_to_idx:
            skipped += 1
            print(f"  [{i+1}/{len(tweets)}] SKIP post_id={post_id} — not found in MemeCap")
            continue

        # Skip if the original meme itself is excluded
        if post_id in excluded_ids:
            skipped_excluded += 1
            print(f"  [{i+1}/{len(tweets)}] EXCLUDED post_id={post_id} — flagged content")
            continue

        target_idx = post_id_to_idx[post_id]
        semantic_idxs, random_idxs = select_distractors(
            target_idx, sim_matrix, NUM_SEMANTIC, NUM_RANDOM, excluded_idxs, rng
        )

        candidates = []

        orig = memes[target_idx]
        candidates.append({
            "meme_post_id": orig["post_id"],
            "image_url": orig.get("url", ""),
            "title": to_str(orig.get("title", "")),
            "selection_method": "original",
        })

        semantic_titles = []
        for idx in semantic_idxs:
            m = memes[idx]
            candidates.append({
                "meme_post_id": m["post_id"],
                "image_url": m.get("url", ""),
                "title": to_str(m.get("title", "")),
                "selection_method": "semantic",
            })
            semantic_titles.append(to_str(m.get("title", "?")))

        for idx in random_idxs:
            m = memes[idx]
            candidates.append({
                "meme_post_id": m["post_id"],
                "image_url": m.get("url", ""),
                "title": to_str(m.get("title", "")),
                "selection_method": "random",
            })

        rng.shuffle(candidates)

        tasks.append({
            "task_id": post_id,
            "tweet_text": tweet["tweet_text"],
            "post_id": post_id,
            "candidates": candidates,
        })

        tweet_preview = tweet["tweet_text"][:60] + ("..." if len(tweet["tweet_text"]) > 60 else "")
        print(f"  [{i+1}/{len(tweets)}] post_id={post_id}")
        print(f"    Tweet: {tweet_preview}")
        print(f"    Original meme: \"{to_str(orig.get('title', '?'))}\"")
        print(f"    Semantic distractors: {semantic_titles}")

        if (i + 1) % 100 == 0:
            print(f"  --- Progress: {i+1}/{len(tweets)} processed, {skipped} not found, {skipped_excluded} excluded ---")

    # 5. Save
    print(f"\n[Step 5/6] Saving to {OUTPUT_FILE} ...")
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
            "skipped_excluded": skipped_excluded,
            "excluded_memes": len(excluded_ids),
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Tasks created: {len(tasks)}")
    print(f"  Skipped (not in MemeCap): {skipped}")
    print(f"  Skipped (excluded/offensive): {skipped_excluded}")
    print(f"  Excluded memes in blocklist: {len(excluded_ids)}")
    print(f"  Candidates per task: {TOTAL_CANDIDATES}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
