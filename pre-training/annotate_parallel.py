#!/usr/bin/env python3
"""
Parallel multi-model VLM annotation pipeline.

Same logic as annotate_with_models.py but uses concurrent requests
for ~5x speedup. Resumable — safe to re-run.

Usage:
    python annotate_parallel.py                  # default $7.50 budget, 10 workers
    python annotate_parallel.py --budget 5.00
    python annotate_parallel.py --workers 20
    python annotate_parallel.py --dry-run
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
import base64
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    {
        "id": "bytedance-seed/seed-1.6-flash",
        "annotator_id": "model_1_seed_flash",
        "cost_per_1m_input": 0.075,
        "cost_per_1m_output": 0.30,
    },
    {
        "id": "google/gemini-3.1-flash-lite-preview",
        "annotator_id": "model_2_gemini_lite",
        "cost_per_1m_input": 0.25,
        "cost_per_1m_output": 1.50,
    },
    {
        "id": "google/gemini-3-flash-preview",
        "annotator_id": "model_3_gemini_flash",
        "cost_per_1m_input": 0.50,
        "cost_per_1m_output": 3.00,
    },
]

RANKINGS_CSV = "meme_rankings.csv"

DEFAULT_BUDGET_USD = 7.50
ANNOTATIONS_REQUIRED = 3
DEFAULT_WORKERS = 10
MAX_RETRIES = 2

INPUT_CSV = "annotations.csv"
OUTPUT_CSV = "annotations_augmented.csv"
TASKS_FILE = "annotation_tasks_clean.json"
HTTP_REFERER = "http://localhost"
APP_TITLE = "memecap-model-annotator"

EST_INPUT_TOKENS = 1500
EST_OUTPUT_TOKENS = 50

CSV_FIELDS = [
    "annotation_id", "task_id", "candidate_index", "tweet_text",
    "meme_post_id", "image_url", "meme_title", "selection_method",
    "annotator_id", "is_good_reply", "flag_inappropriate", "created_at",
]


# =========================
# HELPERS
# =========================

def check_image_accessible(url: str) -> bool:
    try:
        r = requests.head(url, timeout=3, allow_redirects=True)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "")
            if "image" in ct or "octet-stream" in ct:
                return True
        r = requests.get(url, timeout=3, headers={"Range": "bytes=0-1023"},
                         stream=True)
        return r.status_code in (200, 206)
    except Exception:
        return False


def download_image_as_data_url(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "image/jpeg")
        b64 = base64.b64encode(r.content).decode("utf-8")
        return f"data:{ct};base64,{b64}"
    except Exception:
        return None


def build_prompt(tweet_text: str, meme_title: str) -> str:
    return f"""Imagine you're scrolling Twitter/X and you see this tweet:

"{tweet_text}"

Someone replies with the meme image attached (titled: "{meme_title}").

Would this meme reply make you laugh, nod, or think "good one"? Meme replies don't need to be literal — they can be sarcastic, ironic, exaggerated, or loosely related as long as the vibe fits.

Say YES if:
- The meme is a "good enough" reply — it doesn't have to be the best or perfect
- There's some connection, even if loose or vibes-based
- You could imagine someone posting this as a reply and it wouldn't feel out of place

Say NO if:
- The meme has nothing to do with the tweet at all
- It would feel completely random or confusing as a reply

Also flag if the meme contains inappropriate or offensive content.

Respond with ONLY a JSON object, no other text:
{{"reply": "yes" or "no", "flag": "yes" or "no"}}"""


def parse_response(text: str) -> Tuple[int, int]:
    text = text.strip()
    try:
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            data = json.loads(json_match.group())
            reply = str(data.get("reply", "no")).lower().strip()
            flag = str(data.get("flag", "no")).lower().strip()
            if reply == "broken":
                return (-1, 0)
            is_good = 1 if reply in ("yes", "1", "true") else 0
            is_flag = 1 if flag in ("yes", "1", "true") else 0
            return (is_good, is_flag)
    except (json.JSONDecodeError, AttributeError):
        pass
    lower = text.lower()
    if "broken" in lower:
        return (-1, 0)
    is_good = 1 if re.search(r'\byes\b', lower) else 0
    is_flag = 1 if ("inappropriate" in lower or "offensive" in lower) else 0
    return (is_good, is_flag)


def estimate_call_cost(model: dict) -> float:
    inp = (EST_INPUT_TOKENS / 1_000_000) * model["cost_per_1m_input"]
    out = (EST_OUTPUT_TOKENS / 1_000_000) * model["cost_per_1m_output"]
    return inp + out


def call_openrouter(model_id: str, prompt: str, image_url: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": APP_TITLE,
    }
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.3,
        "max_tokens": 60,
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers,
                              json=payload, timeout=120)
            if r.status_code >= 400 and attempt == 0:
                data_url = download_image_as_data_url(image_url)
                if data_url:
                    content[-1] = {"type": "image_url", "image_url": {"url": data_url}}
                    payload["messages"][0]["content"] = content
                    r = requests.post(OPENROUTER_URL, headers=headers,
                                      json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            c = data["choices"][0]["message"]["content"]
            if not c:
                raise ValueError(f"Empty response from {model_id}")
            return c.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# =========================
# LOAD
# =========================

def load_existing_annotations(csv_path: str) -> Tuple[int, Dict[str, set]]:
    max_id = 0
    item_annotators: Dict[str, set] = defaultdict(set)
    if not Path(csv_path).exists():
        return max_id, item_annotators
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = int(row["annotation_id"])
            if aid > max_id:
                max_id = aid
            key = (row["task_id"], row["candidate_index"])
            item_annotators[key].add(row["annotator_id"])
    return max_id, item_annotators


def load_all_items(tasks_file: str) -> List[dict]:
    with open(tasks_file, encoding="utf-8") as f:
        data = json.load(f)
    tasks = data.get("tasks", [])
    random.seed(42)
    random.shuffle(tasks)
    items = []
    for ti, task in enumerate(tasks):
        task_id = f"{task['task_id']}_{ti}"
        tweet_text = task["tweet_text"]
        for ci, cand in enumerate(task["candidates"]):
            items.append({
                "task_id": task_id,
                "candidate_index": str(ci),
                "tweet_text": tweet_text,
                "meme_post_id": cand["meme_post_id"],
                "image_url": cand["image_url"],
                "meme_title": cand["title"],
                "selection_method": cand["selection_method"],
            })
    return items


def generate_rankings(csv_path: str):
    print(f"\n{'='*50}")
    print(f"GENERATING RANKINGS")
    print(f"{'='*50}")

    item_data: Dict[Tuple[str, str], dict] = {}
    item_votes: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["task_id"], row["candidate_index"])
            vote = int(row["is_good_reply"])
            item_votes[key].append(vote)
            if key not in item_data:
                item_data[key] = {
                    "task_id": row["task_id"],
                    "candidate_index": row["candidate_index"],
                    "tweet_text": row["tweet_text"],
                    "meme_post_id": row["meme_post_id"],
                    "image_url": row["image_url"],
                    "meme_title": row["meme_title"],
                    "selection_method": row["selection_method"],
                }

    task_items: Dict[str, List[dict]] = defaultdict(list)
    for key, data in item_data.items():
        votes = item_votes[key]
        real_votes = [v for v in votes if v != -1]
        entry = dict(data)
        entry["avg_score"] = round(sum(real_votes) / len(real_votes), 4) if real_votes else 0.0
        entry["num_votes"] = len(real_votes)
        entry["num_yes"] = sum(1 for v in real_votes if v == 1)
        entry["num_no"] = sum(1 for v in real_votes if v == 0)
        entry["num_broken"] = sum(1 for v in votes if v == -1)
        task_items[data["task_id"]].append(entry)

    ranking_fields = [
        "task_id", "tweet_text", "rank", "candidate_index",
        "meme_post_id", "image_url", "meme_title", "selection_method",
        "avg_score", "num_votes", "num_yes", "num_no", "num_broken",
    ]
    ranked_rows = []
    for task_id in sorted(task_items.keys()):
        candidates = task_items[task_id]
        candidates.sort(key=lambda x: (-x["avg_score"], x["candidate_index"]))
        for rank, entry in enumerate(candidates, 1):
            entry["rank"] = rank
            ranked_rows.append(entry)

    with open(RANKINGS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ranking_fields)
        writer.writeheader()
        writer.writerows(ranked_rows)

    tasks_ranked = len(task_items)
    items_with_votes = sum(1 for r in ranked_rows if r["num_votes"] > 0)
    avg_votes = sum(r["num_votes"] for r in ranked_rows) / len(ranked_rows) if ranked_rows else 0
    print(f"  Tasks ranked: {tasks_ranked}")
    print(f"  Items with votes: {items_with_votes}")
    print(f"  Avg votes per item: {avg_votes:.1f}")
    print(f"  Output: {RANKINGS_CSV}")


# =========================
# PARALLEL WORKER
# =========================

def annotate_one(item: dict, model: dict, image_url: str) -> dict:
    """
    Annotate a single item with a single model. Returns a result dict.
    Called from thread pool — must be thread-safe (no shared mutable state).
    """
    prompt = build_prompt(item["tweet_text"], item["meme_title"])
    try:
        response = call_openrouter(
            model_id=model["id"],
            prompt=prompt,
            image_url=image_url,
        )
        is_good_reply, flag_inappropriate = parse_response(response)
        error = None
    except Exception as e:
        is_good_reply, flag_inappropriate = -1, 0
        error = str(e)

    return {
        "item": item,
        "model": model,
        "is_good_reply": is_good_reply,
        "flag_inappropriate": flag_inappropriate,
        "error": error,
    }


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Parallel VLM annotation pipeline")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                        help=f"Budget cap in USD (default: {DEFAULT_BUDGET_USD})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Concurrent workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show work queue stats without making API calls")
    args = parser.parse_args()

    budget = args.budget
    num_workers = args.workers

    if not OPENROUTER_API_KEY:
        print("Error: set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    # ── 1. Prepare output CSV ──────────────────────────────────
    output_path = Path(OUTPUT_CSV)
    if not output_path.exists():
        if Path(INPUT_CSV).exists():
            shutil.copy2(INPUT_CSV, OUTPUT_CSV)
            print(f"Copied {INPUT_CSV} -> {OUTPUT_CSV}")
        else:
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
            print(f"Created empty {OUTPUT_CSV}")

    # ── 2. Load existing state ─────────────────────────────────
    print("Loading existing annotations...")
    max_ann_id, item_annotators = load_existing_annotations(OUTPUT_CSV)
    total_existing = sum(len(v) for v in item_annotators.values())
    print(f"  {total_existing} annotations across {len(item_annotators)} items")

    # ── 3. Load all items ──────────────────────────────────────
    print("Loading annotation tasks...")
    all_items = load_all_items(TASKS_FILE)
    print(f"  {len(all_items)} total items ({len(all_items) // 10} tasks x 10 candidates)")

    # ── 4. Pre-check image accessibility ───────────────────────
    print("Checking image accessibility...")
    unique_urls = set(item["image_url"] for item in all_items)
    image_ok_cache: Dict[str, bool] = {}

    def check_url(url):
        return url, check_image_accessible(url)

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(check_url, url): url for url in unique_urls}
        done_count = 0
        for future in as_completed(futures):
            url, ok = future.result()
            image_ok_cache[url] = ok
            done_count += 1
            if done_count % 200 == 0:
                print(f"  Checked {done_count}/{len(unique_urls)} URLs...")

    broken_urls = sum(1 for v in image_ok_cache.values() if not v)
    print(f"  {len(unique_urls)} unique URLs: {len(unique_urls) - broken_urls} ok, {broken_urls} broken")

    # ── 5. Build work queue ────────────────────────────────────
    # Each entry is (item, model) — a single API call
    work_queue: List[Tuple[dict, dict]] = []

    for item in all_items:
        # Skip broken images
        if not image_ok_cache.get(item["image_url"], False):
            continue

        key = (item["task_id"], item["candidate_index"])
        existing = item_annotators.get(key, set())

        if len(existing) >= ANNOTATIONS_REQUIRED:
            continue

        num_needed = ANNOTATIONS_REQUIRED - len(existing)
        has_human = any(not a.startswith("model_") for a in existing)

        available = [m for m in MODELS if m["annotator_id"] not in existing]
        models_to_call = available[:num_needed]

        for model in models_to_call:
            work_queue.append((item, model, has_human))

    # Sort: human-annotated items first
    work_queue.sort(key=lambda x: (not x[2],))

    # Trim to budget
    trimmed_queue = []
    running_est = 0.0
    for item, model, has_human in work_queue:
        cost = estimate_call_cost(model)
        if running_est + cost > budget:
            break
        running_est += cost
        trimmed_queue.append((item, model))

    total_calls = len(trimmed_queue)
    est_cost = running_est

    print(f"\n{'='*50}")
    print(f"WORK QUEUE SUMMARY")
    print(f"{'='*50}")
    print(f"  Total calls (pre-budget):  {len(work_queue)}")
    print(f"  Calls within budget:       {total_calls}")
    print(f"  Estimated cost:            ${est_cost:.2f}")
    print(f"  Budget limit:              ${budget:.2f}")
    print(f"  Workers:                   {num_workers}")
    print(f"  Broken images skipped:     {broken_urls}")

    est_time = total_calls * 1.5 / num_workers
    print(f"  Estimated time:            ~{est_time/60:.0f} min")

    print(f"\n  Model cost breakdown:")
    for m in MODELS:
        c = estimate_call_cost(m)
        print(f"    {m['annotator_id']:30s} ~${c*1000:.3f}/1k calls")

    if args.dry_run:
        print("\n[DRY RUN] No API calls made.")
        return

    if total_calls == 0:
        print("\nNothing to do.")
        generate_rankings(OUTPUT_CSV)
        return

    print(f"\nStarting parallel annotation ({num_workers} workers)...")
    print(f"{'='*50}\n")

    # ── 6. Run parallel annotations ───────────────────────────
    next_ann_id = max_ann_id + 1
    ann_id_lock = threading.Lock()
    write_lock = threading.Lock()
    stats = {"done": 0, "yes": 0, "no": 0, "broken": 0, "errors": 0, "cost": 0.0}

    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)

    def process_result(result, model):
        nonlocal next_ann_id
        item = result["item"]
        is_good = result["is_good_reply"]
        flag = result["flag_inappropriate"]

        with ann_id_lock:
            ann_id = next_ann_id
            next_ann_id += 1

        row = {
            "annotation_id": ann_id,
            "task_id": item["task_id"],
            "candidate_index": item["candidate_index"],
            "tweet_text": item["tweet_text"],
            "meme_post_id": item["meme_post_id"],
            "image_url": item["image_url"],
            "meme_title": item["meme_title"],
            "selection_method": item["selection_method"],
            "annotator_id": model["annotator_id"],
            "is_good_reply": is_good,
            "flag_inappropriate": flag,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with write_lock:
            writer.writerow(row)
            csv_file.flush()

            cost = estimate_call_cost(model)
            stats["cost"] += cost
            stats["done"] += 1
            if is_good == 1:
                stats["yes"] += 1
            elif is_good == 0:
                stats["no"] += 1
            else:
                stats["broken"] += 1
            if result["error"]:
                stats["errors"] += 1

            if stats["done"] % 100 == 0:
                pct = stats["done"] / total_calls * 100
                print(
                    f"  [{stats['done']:,}/{total_calls:,}] {pct:.1f}%  "
                    f"${stats['cost']:.2f}/{budget:.2f}  "
                    f"yes={stats['yes']} no={stats['no']} err={stats['errors']}"
                )

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {}
            for item, model in trimmed_queue:
                future = pool.submit(
                    annotate_one, item, model, item["image_url"]
                )
                futures[future] = model

            for future in as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                    process_result(result, model)
                except Exception as e:
                    print(f"  [FATAL] {e}")
                    stats["errors"] += 1
                    stats["done"] += 1

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! {stats['done']} annotations saved.")
    finally:
        csv_file.close()

    # ── 7. Summary ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"  Annotations added: {stats['done']}")
    print(f"  Yes: {stats['yes']}  No: {stats['no']}  Broken: {stats['broken']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Estimated cost: ~${stats['cost']:.2f}")
    print(f"  Output: {OUTPUT_CSV}")

    # ── 8. Generate rankings ──────────────────────────────────
    generate_rankings(OUTPUT_CSV)


if __name__ == "__main__":
    main()
