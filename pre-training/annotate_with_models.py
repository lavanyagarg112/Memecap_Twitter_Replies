#!/usr/bin/env python3
"""
Multi-model VLM annotation pipeline.

Uses 3 vision-language models as simulated annotators (cheapest first).
Each meme-tweet pair gets up to 3 annotations total (human + model).
Items with existing human annotations are completed first.

Features:
- Copies annotations.csv -> annotations_augmented.csv, appends model annotations
- Budget cap with running cost tracker
- Broken images skipped entirely (no API call wasted)
- Resumable: re-run safely, already-annotated items are skipped

Usage:
    pip install requests python-dotenv
    export OPENROUTER_API_KEY=your_key
    python annotate_with_models.py

    # Or with custom budget:
    python annotate_with_models.py --budget 5.00
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# 3 vision-capable models, cheapest -> most expensive.
# For a given item needing N annotations, the cheapest N models are used.
# Verify pricing at https://openrouter.ai/models
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

DEFAULT_BUDGET_USD = 7.50  # ~$10 SGD
ANNOTATIONS_REQUIRED = 3
SLEEP_BETWEEN_CALLS = 0.1
MAX_RETRIES = 2

INPUT_CSV = "annotations.csv"
OUTPUT_CSV = "annotations_augmented.csv"
TASKS_FILE = "annotation_tasks_clean.json"
HTTP_REFERER = "http://localhost"
APP_TITLE = "memecap-model-annotator"

# Rough token estimates per call (for budget tracking)
EST_INPUT_TOKENS = 1500  # prompt + image
EST_OUTPUT_TOKENS = 50   # JSON response

CSV_FIELDS = [
    "annotation_id", "task_id", "candidate_index", "tweet_text",
    "meme_post_id", "image_url", "meme_title", "selection_method",
    "annotator_id", "is_good_reply", "flag_inappropriate", "created_at",
]


# =========================
# HELPERS
# =========================

def check_image_accessible(url: str) -> bool:
    """Return True if the image URL resolves to a valid image."""
    try:
        r = requests.head(url, timeout=3, allow_redirects=True)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "")
            if "image" in ct or "octet-stream" in ct:
                return True
        # Some servers reject HEAD; try a small GET
        r = requests.get(url, timeout=3, headers={"Range": "bytes=0-1023"},
                         stream=True)
        return r.status_code in (200, 206)
    except Exception:
        return False


def download_image_as_data_url(url: str) -> Optional[str]:
    """Download image and return as base64 data URL for API fallback."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "image/jpeg")
        b64 = base64.b64encode(r.content).decode("utf-8")
        return f"data:{ct};base64,{b64}"
    except Exception:
        return None




def build_prompt_with_image(tweet_text: str, meme_title: str) -> str:
    """Prompt when the meme image is available."""
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




def parse_model_response(text: str) -> Tuple[int, int]:
    """
    Parse model JSON response into (is_good_reply, flag_inappropriate).
    Returns (-1, 0) for broken images or unparseable responses.
    """
    text = text.strip()

    # Try to extract JSON
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

    # Fallback heuristics
    lower = text.lower()
    if "broken" in lower:
        return (-1, 0)

    is_good = 1 if re.search(r'\byes\b', lower) else 0
    is_flag = 1 if ("inappropriate" in lower or "offensive" in lower) else 0
    return (is_good, is_flag)


def estimate_call_cost(model: dict) -> float:
    """Estimate USD cost of one API call."""
    inp = (EST_INPUT_TOKENS / 1_000_000) * model["cost_per_1m_input"]
    out = (EST_OUTPUT_TOKENS / 1_000_000) * model["cost_per_1m_output"]
    return inp + out


def call_openrouter(model_id: str, prompt: str,
                    image_url: Optional[str] = None) -> str:
    """Call OpenRouter API. Retries once on failure."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": APP_TITLE,
    }

    content: list = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url},
        })

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

            # If URL-based image fails, try base64 fallback
            if r.status_code >= 400 and image_url and attempt == 0:
                data_url = download_image_as_data_url(image_url)
                if data_url:
                    content[-1] = {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                    payload["messages"][0]["content"] = content
                    r = requests.post(OPENROUTER_URL, headers=headers,
                                      json=payload, timeout=120)

            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise ValueError(f"Empty response from {model_id}")
            return content.strip()

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# =========================
# LOAD / SAVE
# =========================

def load_existing_annotations(csv_path: str) -> Tuple[int, Dict[str, set]]:
    """
    Load CSV and return (max_annotation_id, item_annotators).
    item_annotators maps (task_id, candidate_index) -> set of annotator_ids.
    """
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


def load_human_labels(csv_path: str) -> Dict[Tuple[str, str], dict]:
    """
    Load human annotations and compute majority vote per item.
    Returns {(task_id, candidate_index): {
        "votes": [list of 0/1], "majority": 0 or 1, "count": N,
        "tweet_text": str, "meme_post_id": str, "image_url": str,
        "meme_title": str, "selection_method": str
    }}
    Only includes items with at least one real (non-skip) human vote.
    """
    item_votes: Dict[Tuple[str, str], dict] = {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip model annotations
            if row["annotator_id"].startswith("model_"):
                continue
            vote = int(row["is_good_reply"])
            if vote == -1:  # skip/broken
                continue

            key = (row["task_id"], row["candidate_index"])
            if key not in item_votes:
                item_votes[key] = {
                    "votes": [],
                    "tweet_text": row["tweet_text"],
                    "meme_post_id": row["meme_post_id"],
                    "image_url": row["image_url"],
                    "meme_title": row["meme_title"],
                    "selection_method": row["selection_method"],
                    "task_id": row["task_id"],
                    "candidate_index": row["candidate_index"],
                }
            item_votes[key]["votes"].append(vote)

    # Compute majority vote
    for key, info in item_votes.items():
        votes = info["votes"]
        info["count"] = len(votes)
        info["majority"] = 1 if sum(votes) > len(votes) / 2 else 0

    return item_votes


def load_all_items(tasks_file: str) -> List[dict]:
    """
    Load annotation_tasks_clean.json and reconstruct items
    with the same shuffled task_id format as app.py.
    """
    with open(tasks_file, encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])

    # Replicate app.py's shuffle
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


# =========================
# VALIDATION
# =========================

def run_validation(budget: float, sample_size: int = 0):
    """
    Run all 5 models on human-annotated items only.
    Compare model predictions to human majority vote.
    Writes results to validation_results.csv and prints agreement report.
    """
    if not OPENROUTER_API_KEY:
        print("Error: set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    VALIDATION_CSV = "validation_results.csv"

    # ── 1. Load human labels ──────────────────────────────────
    print("Loading human annotations...")
    human_labels = load_human_labels(INPUT_CSV)
    print(f"  {len(human_labels)} items with human votes")

    # Sample if requested
    if sample_size > 0 and sample_size < len(human_labels):
        random.seed(42)
        sampled_keys = random.sample(list(human_labels.keys()), sample_size)
        human_labels = {k: human_labels[k] for k in sampled_keys}
        print(f"  Sampled {sample_size} items for validation")

    yes_items = sum(1 for v in human_labels.values() if v["majority"] == 1)
    no_items = len(human_labels) - yes_items
    print(f"  Human majority: {yes_items} yes ({yes_items/len(human_labels)*100:.0f}%), "
          f"{no_items} no ({no_items/len(human_labels)*100:.0f}%)")

    # ── 2. Check for already-completed validation ──────────────
    existing_results: Dict[Tuple[str, str], set] = defaultdict(set)
    if Path(VALIDATION_CSV).exists():
        with open(VALIDATION_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["task_id"], row["candidate_index"])
                existing_results[key].add(row["annotator_id"])
        print(f"  Resuming: {sum(len(v) for v in existing_results.values())} "
              f"existing validation results")

    # ── 4. Estimate cost ──────────────────────────────────────
    total_calls = 0
    for key, info in human_labels.items():
        done = existing_results.get(key, set())
        for m in MODELS:
            if m["annotator_id"] not in done:
                total_calls += 1

    est_cost = sum(estimate_call_cost(m) for m in MODELS) * (total_calls / len(MODELS))
    print(f"\n  Remaining API calls: {total_calls}")
    print(f"  Estimated cost: ~${est_cost:.2f}")
    print(f"  Budget: ${budget:.2f}")

    if total_calls == 0:
        print("\n  All validation calls already completed!")
    else:
        print(f"\nRunning validation...")

    # ── 5. Run models on human-labeled items ──────────────────
    val_fields = [
        "task_id", "candidate_index", "tweet_text", "meme_post_id",
        "image_url", "meme_title", "selection_method",
        "annotator_id", "model_prediction", "human_majority",
        "human_votes", "human_count",
    ]

    write_header = not Path(VALIDATION_CSV).exists()
    image_ok_cache: Dict[str, bool] = {}
    running_cost = 0.0
    calls_done = 0

    with open(VALIDATION_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=val_fields)
        if write_header:
            writer.writeheader()

        items = sorted(human_labels.items())
        for qi, (key, info) in enumerate(items):
            done_models = existing_results.get(key, set())
            models_needed = [m for m in MODELS if m["annotator_id"] not in done_models]
            if not models_needed:
                continue

            img_url = info["image_url"]
            if img_url not in image_ok_cache:
                image_ok_cache[img_url] = check_image_accessible(img_url)
            image_ok = image_ok_cache[img_url]

            if not image_ok:
                continue

            for model in models_needed:
                call_cost = estimate_call_cost(model)
                if running_cost + call_cost > budget:
                    print(f"\nBudget limit reached (${running_cost:.2f})")
                    break

                prompt = build_prompt_with_image(
                    info["tweet_text"], info["meme_title"]
                )

                try:
                    response = call_openrouter(
                        model_id=model["id"],
                        prompt=prompt,
                        image_url=img_url,
                    )
                    prediction, _ = parse_model_response(response)
                except Exception as e:
                    print(f"  [ERR] {model['annotator_id']}: {e}")
                    prediction = -1

                writer.writerow({
                    "task_id": info["task_id"],
                    "candidate_index": info["candidate_index"],
                    "tweet_text": info["tweet_text"],
                    "meme_post_id": info["meme_post_id"],
                    "image_url": info["image_url"],
                    "meme_title": info["meme_title"],
                    "selection_method": info["selection_method"],
                    "annotator_id": model["annotator_id"],
                    "model_prediction": prediction,
                    "human_majority": info["majority"],
                    "human_votes": json.dumps(info["votes"]),
                    "human_count": info["count"],
                })
                f.flush()

                running_cost += call_cost
                calls_done += 1

                if calls_done % 50 == 0:
                    print(f"  [{calls_done}/{total_calls}] ${running_cost:.2f}")

                time.sleep(SLEEP_BETWEEN_CALLS)
            else:
                continue
            break  # budget exceeded

    # ── 6. Report agreement ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  Calls made: {calls_done}  |  Cost: ~${running_cost:.2f}")
    print(f"  Results: {VALIDATION_CSV}\n")

    # Load all results for analysis
    model_stats: Dict[str, dict] = {}
    for m in MODELS:
        model_stats[m["annotator_id"]] = {
            "correct": 0, "total": 0, "yes": 0, "no": 0, "broken": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        }

    if not Path(VALIDATION_CSV).exists():
        return

    with open(VALIDATION_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["annotator_id"]
            if mid not in model_stats:
                continue
            pred = int(row["model_prediction"])
            human = int(row["human_majority"])

            if pred == -1:  # broken/error
                model_stats[mid]["broken"] += 1
                continue

            s = model_stats[mid]
            s["total"] += 1
            if pred == 1:
                s["yes"] += 1
            else:
                s["no"] += 1

            if pred == human:
                s["correct"] += 1

            if pred == 1 and human == 1:
                s["tp"] += 1
            elif pred == 1 and human == 0:
                s["fp"] += 1
            elif pred == 0 and human == 0:
                s["tn"] += 1
            elif pred == 0 and human == 1:
                s["fn"] += 1

    # Print per-model stats
    print(f"  {'Model':<30s} {'Acc':>6s} {'Yes%':>6s} {'Prec':>6s} "
          f"{'Rec':>6s} {'N':>5s} {'Broken':>6s}")
    print(f"  {'-'*29} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*6}")

    for m in MODELS:
        mid = m["annotator_id"]
        s = model_stats[mid]
        if s["total"] == 0:
            print(f"  {mid:<30s} {'n/a':>6s} {'n/a':>6s} {'n/a':>6s} "
                  f"{'n/a':>6s} {0:>5d} {s['broken']:>6d}")
            continue

        acc = s["correct"] / s["total"] * 100
        yes_rate = s["yes"] / s["total"] * 100
        prec = s["tp"] / (s["tp"] + s["fp"]) * 100 if (s["tp"] + s["fp"]) > 0 else 0
        rec = s["tp"] / (s["tp"] + s["fn"]) * 100 if (s["tp"] + s["fn"]) > 0 else 0

        print(f"  {mid:<30s} {acc:>5.1f}% {yes_rate:>5.1f}% {prec:>5.1f}% "
              f"{rec:>5.1f}% {s['total']:>5d} {s['broken']:>6d}")

    # Majority vote across models
    item_model_votes: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    item_human: Dict[Tuple[str, str], int] = {}

    with open(VALIDATION_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = int(row["model_prediction"])
            if pred == -1:
                continue
            key = (row["task_id"], row["candidate_index"])
            item_model_votes[key].append(pred)
            item_human[key] = int(row["human_majority"])

    # Ensemble accuracy (majority of all models)
    ensemble_correct = 0
    ensemble_total = 0
    for key, votes in item_model_votes.items():
        if len(votes) < 2:
            continue
        model_majority = 1 if sum(votes) > len(votes) / 2 else 0
        if model_majority == item_human[key]:
            ensemble_correct += 1
        ensemble_total += 1

    if ensemble_total > 0:
        ens_acc = ensemble_correct / ensemble_total * 100
        print(f"\n  {'ENSEMBLE (majority vote)':<30s} {ens_acc:>5.1f}%"
              f"{'':>6s}{'':>6s}{'':>6s} {ensemble_total:>5d}")

    print(f"\n  Human baseline: {yes_items} yes / {no_items} no "
          f"({yes_items/len(human_labels)*100:.0f}% / {no_items/len(human_labels)*100:.0f}%)")
    print(f"  (A model that always says 'no' would get "
          f"{no_items/len(human_labels)*100:.0f}% accuracy)\n")


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Multi-model VLM annotation pipeline")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                        help=f"Budget cap in USD (default: {DEFAULT_BUDGET_USD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show work queue stats without making API calls")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only annotate human-labeled items, then report agreement")
    parser.add_argument("--validate-sample", type=int, default=0,
                        help="Number of items to sample for validation (0 = all)")
    args = parser.parse_args()

    if args.validate_only:
        run_validation(args.budget, args.validate_sample)
        return

    budget = args.budget

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
            # Create empty CSV with header
            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
            print(f"Created empty {OUTPUT_CSV}")

    # ── 2. Load existing state ─────────────────────────────────
    print("Loading existing annotations...")
    max_ann_id, item_annotators = load_existing_annotations(OUTPUT_CSV)
    total_existing = sum(len(v) for v in item_annotators.values())
    print(f"  {total_existing} annotations across {len(item_annotators)} items (max ID: {max_ann_id})")

    # ── 3. Load all items ──────────────────────────────────────
    print("Loading annotation tasks...")
    all_items = load_all_items(TASKS_FILE)
    print(f"  {len(all_items)} total items ({len(all_items) // 10} tasks x 10 candidates)")

    # ── 4. Build prioritized work queue ─────────────────────────
    work_queue = []  # [(item_dict, [models_to_call], has_human_annotation)]

    for item in all_items:
        key = (item["task_id"], item["candidate_index"])
        existing = item_annotators.get(key, set())

        if len(existing) >= ANNOTATIONS_REQUIRED:
            continue

        num_needed = ANNOTATIONS_REQUIRED - len(existing)
        has_human = any(not a.startswith("model_") for a in existing)

        # Pick cheapest N models that haven't annotated this item yet
        available = [m for m in MODELS if m["annotator_id"] not in existing]
        models_to_call = available[:num_needed]

        if models_to_call:
            work_queue.append((item, models_to_call, has_human))

    # Prioritize: items with human annotations first, then by fewer needed
    work_queue.sort(key=lambda x: (not x[2], len(x[1])))

    total_calls = sum(len(m) for _, m, _ in work_queue)
    est_cost = sum(
        sum(estimate_call_cost(m) for m in models)
        for _, models, _ in work_queue
    )

    print(f"\n{'='*50}")
    print(f"WORK QUEUE SUMMARY")
    print(f"{'='*50}")
    print(f"  Items to annotate:  {len(work_queue)}")
    print(f"  Total API calls:    {total_calls}")
    print(f"  Estimated cost:     ${est_cost:.2f}")
    print(f"  Budget limit:       ${budget:.2f}")

    items_with_human = sum(1 for _, _, h in work_queue if h)
    print(f"  Items w/ human ann: {items_with_human} (prioritized)")

    calls_within_budget = 0
    cost_so_far = 0
    for _, models, _ in work_queue:
        for m in models:
            c = estimate_call_cost(m)
            if cost_so_far + c <= budget:
                calls_within_budget += 1
                cost_so_far += c
    print(f"  Calls within budget: ~{calls_within_budget}")
    print(f"  Items within budget: ~{calls_within_budget // ANNOTATIONS_REQUIRED}")

    print(f"\n  Model cost breakdown:")
    for m in MODELS:
        c = estimate_call_cost(m)
        print(f"    {m['annotator_id']:30s} ~${c*1000:.3f}/1k calls")

    if args.dry_run:
        print("\n[DRY RUN] No API calls made.")
        return

    print(f"\nStarting annotation pipeline...")
    print(f"{'='*50}\n")

    # ── 6. Run annotations ─────────────────────────────────────
    running_cost = 0.0
    next_ann_id = max_ann_id + 1
    total_done = 0
    broken_count = 0
    yes_count = 0
    no_count = 0
    errors = 0

    # Cache image accessibility checks
    image_ok_cache: Dict[str, bool] = {}

    budget_exceeded = False

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        for qi, (item, models_to_call, has_human) in enumerate(work_queue):
            if budget_exceeded:
                break

            img_url = item["image_url"]

            # Check image (cached)
            if img_url not in image_ok_cache:
                image_ok_cache[img_url] = check_image_accessible(img_url)
            image_ok = image_ok_cache[img_url]

            # Skip broken images entirely
            if not image_ok:
                broken_count += 1
                continue

            for model in models_to_call:
                call_cost = estimate_call_cost(model)
                if running_cost + call_cost > budget:
                    print(f"\nBudget limit reached (${running_cost:.2f} / ${budget:.2f})")
                    budget_exceeded = True
                    break

                prompt = build_prompt_with_image(
                    item["tweet_text"], item["meme_title"]
                )

                # Call API
                try:
                    response = call_openrouter(
                        model_id=model["id"],
                        prompt=prompt,
                        image_url=img_url,
                    )
                    is_good_reply, flag_inappropriate = parse_model_response(response)
                except Exception as e:
                    print(f"  [ERR] {model['annotator_id']} on {item['task_id']}/{item['candidate_index']}: {e}")
                    is_good_reply, flag_inappropriate = -1, 0
                    errors += 1

                # Track stats
                if is_good_reply == 1:
                    yes_count += 1
                elif is_good_reply == 0:
                    no_count += 1
                else:
                    broken_count += 1

                # Write row
                row = {
                    "annotation_id": next_ann_id,
                    "task_id": item["task_id"],
                    "candidate_index": item["candidate_index"],
                    "tweet_text": item["tweet_text"],
                    "meme_post_id": item["meme_post_id"],
                    "image_url": item["image_url"],
                    "meme_title": item["meme_title"],
                    "selection_method": item["selection_method"],
                    "annotator_id": model["annotator_id"],
                    "is_good_reply": is_good_reply,
                    "flag_inappropriate": flag_inappropriate,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                writer.writerow(row)
                f.flush()

                next_ann_id += 1
                running_cost += call_cost
                total_done += 1

                # Progress every 100 calls
                if total_done % 100 == 0:
                    pct = (total_done / total_calls * 100) if total_calls else 0
                    print(
                        f"  [{total_done:,}/{total_calls:,}] {pct:.1f}%  "
                        f"${running_cost:.2f}/{budget:.2f}  "
                        f"yes={yes_count} no={no_count} broken={broken_count}"
                    )

                time.sleep(SLEEP_BETWEEN_CALLS)

    # ── 7. Summary ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"  Annotations added: {total_done}")
    print(f"  Yes: {yes_count}  No: {no_count}  Broken: {broken_count}")
    print(f"  Errors: {errors}")
    print(f"  Estimated cost: ~${running_cost:.2f}")
    print(f"  Output: {OUTPUT_CSV}")

    # ── 8. Generate rankings ──────────────────────────────────
    generate_rankings(OUTPUT_CSV)


def generate_rankings(csv_path: str):
    """
    Read the augmented annotations CSV and produce meme_rankings.csv.
    For each tweet (task), rank the 10 candidate memes by average score.

    Output columns:
        task_id, tweet_text, rank, candidate_index, meme_post_id,
        image_url, meme_title, selection_method,
        avg_score, num_votes, num_yes, num_no, num_broken
    """
    print(f"\n{'='*50}")
    print(f"GENERATING RANKINGS")
    print(f"{'='*50}")

    # Collect votes per item
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

    # Group by task
    task_items: Dict[str, List[dict]] = defaultdict(list)

    for key, data in item_data.items():
        votes = item_votes[key]
        real_votes = [v for v in votes if v != -1]
        num_yes = sum(1 for v in real_votes if v == 1)
        num_no = sum(1 for v in real_votes if v == 0)
        num_broken = sum(1 for v in votes if v == -1)
        avg_score = sum(real_votes) / len(real_votes) if real_votes else 0.0

        entry = dict(data)
        entry["avg_score"] = round(avg_score, 4)
        entry["num_votes"] = len(real_votes)
        entry["num_yes"] = num_yes
        entry["num_no"] = num_no
        entry["num_broken"] = num_broken

        task_items[data["task_id"]].append(entry)

    # Sort candidates within each task by avg_score descending
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
    items_ranked = len(ranked_rows)
    items_with_votes = sum(1 for r in ranked_rows if r["num_votes"] > 0)
    avg_votes = sum(r["num_votes"] for r in ranked_rows) / items_ranked if items_ranked else 0

    print(f"  Tasks ranked: {tasks_ranked}")
    print(f"  Items ranked: {items_ranked}")
    print(f"  Items with votes: {items_with_votes}")
    print(f"  Avg votes per item: {avg_votes:.1f}")
    print(f"  Output: {RANKINGS_CSV}")


if __name__ == "__main__":
    main()
