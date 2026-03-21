"""
Clean tweet JSONL files by removing errors.

If a meme (post_id) has an error in ANY file, it gets removed from ALL files
to keep the direct and indirect datasets balanced.

Run:
    python clean_tweets.py
"""

import json
from pathlib import Path

# Files to clean together (must stay balanced)
TWEET_FILES = [
    "train_memecap_tweets.jsonl",
    "train_memecap_tweets_indirect.jsonl",
]

SUFFIX = "_clean"  # output: train_memecap_tweets_clean.jsonl


def is_error_record(rec: dict) -> bool:
    """Check if this record is an actual API/generation error."""
    # Has an explicit "error" key (set by the generation script on failure)
    if "error" in rec.keys():
        return True
    # Missing the tweet field entirely
    if "tweet_that_meme_replies_to" not in rec.keys():
        return True
    # Tweet field is empty or whitespace
    tweet = rec.get("tweet_that_meme_replies_to", "")
    if not tweet or not tweet.strip():
        return True
    return False


def main():
    # 1. Load all files
    data = {}  # fname -> list of records
    for fname in TWEET_FILES:
        path = Path(fname)
        if not path.exists():
            print(f"Warning: {fname} not found, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            data[fname] = [json.loads(line) for line in f]
        print(f"Loaded {len(data[fname])} records from {fname}")

    if not data:
        print("No files found.")
        return

    # 2. Find post_ids with errors in ANY file
    error_ids = set()
    for fname, records in data.items():
        for rec in records:
            if is_error_record(rec):
                error_ids.add(rec["post_id"])
                print(f"  Error in {fname}: post_id={rec['post_id']} — {rec.get('error', 'missing tweet')}")

    print(f"\nFound {len(error_ids)} post_ids with errors across all files.")

    # 3. Write cleaned files
    for fname, records in data.items():
        clean = [r for r in records if r["post_id"] not in error_ids]
        removed = len(records) - len(clean)

        out_path = Path(fname).stem + SUFFIX + ".jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in clean:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"{fname}: {len(records)} -> {len(clean)} ({removed} removed) -> {out_path}")


if __name__ == "__main__":
    main()
