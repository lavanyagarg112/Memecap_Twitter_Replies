"""
Interactive meme content flagger.

Scans all memes for potentially offensive words, shows each
flagged meme to you for review. You decide which ones to exclude.

Saves a blocklist to flagged_memes.json that select_candidates_v2.py reads.

Run:
    python flag_memes.py
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import requests


# =========================
# CONFIG
# =========================

DATASET_SPLIT = "trainval"
DATASET_URLS = {
    "test": "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-test.json",
    "trainval": "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-trainval.json",
}

OUTPUT_FILE = "flagged_memes.json"

# Words to scan for — catches variations via substring matching
FLAGGED_WORDS = [
    # sexual
    "sex", "sexual", "sexy", "nude", "naked", "porn", "erotic",
    "orgasm", "masturbat", "genital", "penis", "vagina", "boob",
    "tits", "dick", "cock", "pussy", "boner", "horny", "cum",
    "blowjob", "handjob", "dildo", "fetish", "hentai", "nsfw",
    "stripper", "prostitut", "whore", "slut",
    # slurs
    "nigger", "nigga", "faggot", "fag", "retard", "tranny",
    "chink", "spic", "kike", "coon", "wetback", "dyke",
    # violence/extreme
    "rape", "rapist", "molest", "pedophil", "pedo",
]

_FLAG_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FLAGGED_WORDS) + r")",
    re.IGNORECASE,
)


# =========================
# HELPERS
# =========================

def safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def get_all_text(meme: Dict) -> str:
    parts = [safe_str(meme.get("title", ""))]
    for cap in meme.get("img_captions", []):
        parts.append(safe_str(cap))
    for cap in meme.get("meme_captions", []):
        parts.append(safe_str(cap))
    for m in meme.get("metaphors", []):
        if isinstance(m, dict):
            parts.append(safe_str(m.get("metaphor", "")))
            parts.append(safe_str(m.get("meaning", "")))
    return " ".join(parts)


def check_flagged(text: str) -> List[str]:
    return list(set(_FLAG_PATTERN.findall(text.lower())))


# =========================
# MAIN
# =========================

def main():
    # Load existing blocklist if resuming
    existing = {}
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
            existing = {e["post_id"]: e["excluded"] for e in data.get("reviewed", [])}
        print(f"Loaded {len(existing)} previously reviewed memes from {OUTPUT_FILE}")

    # Fetch dataset
    url = DATASET_URLS[DATASET_SPLIT]
    print(f"Downloading MemeCap ({DATASET_SPLIT}) ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    memes = r.json()
    print(f"Loaded {len(memes)} memes.\n")

    # Scan
    flagged = []
    for m in memes:
        text = get_all_text(m)
        matches = check_flagged(text)
        if matches:
            flagged.append((m, matches))

    print(f"Found {len(flagged)} memes with potentially offensive words.")
    to_review = [(m, matches) for m, matches in flagged if m["post_id"] not in existing]
    print(f"{len(to_review)} need review ({len(existing)} already reviewed).\n")

    if not to_review:
        print("Nothing new to review!")
        save_results(existing, flagged)
        return

    # Interactive review
    reviewed = dict(existing)  # copy
    print("For each meme, type 'y' to EXCLUDE it, 'n' to keep it, or 'q' to quit.\n")
    print("=" * 60)

    for i, (meme, matches) in enumerate(to_review):
        pid = meme["post_id"]
        title = safe_str(meme.get("title", ""))
        captions = meme.get("meme_captions", [])

        print(f"\n[{i+1}/{len(to_review)}] post_id: {pid}")
        print(f"  Title: {title}")
        print(f"  Matched words: {matches}")
        if captions:
            print(f"  Meme captions:")
            for cap in captions:
                print(f"    - {safe_str(cap)}")
        print(f"  Image: {meme.get('url', 'N/A')}")

        while True:
            choice = input("\n  Exclude? (y/n/q): ").strip().lower()
            if choice in ("y", "n", "q"):
                break
            print("  Please enter y, n, or q.")

        if choice == "q":
            print("\nSaving progress and quitting...")
            break

        reviewed[pid] = (choice == "y")
        if choice == "y":
            print(f"  -> EXCLUDED")
        else:
            print(f"  -> Kept")

    save_results(reviewed, flagged)


def save_results(reviewed: Dict[str, bool], flagged: list):
    """Save review results."""
    reviewed_list = []
    for m, matches in flagged:
        pid = m["post_id"]
        reviewed_list.append({
            "post_id": pid,
            "title": safe_str(m.get("title", "")),
            "matched_words": matches,
            "excluded": reviewed.get(pid, None),  # None = not yet reviewed
        })

    excluded = [r for r in reviewed_list if r["excluded"] is True]
    kept = [r for r in reviewed_list if r["excluded"] is False]
    pending = [r for r in reviewed_list if r["excluded"] is None]

    output = {
        "reviewed": reviewed_list,
        "excluded_post_ids": [r["post_id"] for r in excluded],
        "summary": {
            "total_flagged": len(reviewed_list),
            "excluded": len(excluded),
            "kept": len(kept),
            "pending_review": len(pending),
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"  Excluded: {len(excluded)}")
    print(f"  Kept: {len(kept)}")
    print(f"  Pending: {len(pending)}")


if __name__ == "__main__":
    main()
