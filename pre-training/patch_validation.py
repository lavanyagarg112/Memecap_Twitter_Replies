#!/usr/bin/env python3
"""
Patch validation_results.csv:
1. Keep seed-flash and gemini-lite results
2. Remove seed-2.0-mini, mistral, gpt-nano results
3. Run 3 new models (gemini-3-flash, kimi-k2.5, qwen3.5-27b) on same items
4. Print agreement report for all 5

Usage:
    python patch_validation.py
"""

import csv
import json
import os
import re
import time
import base64
from collections import defaultdict
from typing import Optional

import requests
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VALIDATION_CSV = "validation_results.csv"

NEW_MODELS = [
    {"id": "google/gemini-3-flash-preview", "annotator_id": "model_3_gemini_flash"},
    {"id": "bytedance-seed/seed-2.0-lite", "annotator_id": "model_4_seed_lite"},
    {"id": "google/gemma-3-27b-it", "annotator_id": "model_5_gemma"},
]

# Models to remove
REMOVE_IDS = {"model_2_seed", "model_3_mistral", "model_3_reka", "model_4_gpt_nano", "model_4_kimi", "model_4_qwen", "model_5_mimo"}

HTTP_REFERER = "http://localhost"
APP_TITLE = "memecap-model-annotator"
MEMECAP_URL = "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-trainval.json"
SLEEP = 0.1
MAX_RETRIES = 2


def check_image_accessible(url):
    try:
        r = requests.head(url, timeout=3, allow_redirects=True)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "")
            if "image" in ct or "octet-stream" in ct:
                return True
        r = requests.get(url, timeout=3, headers={"Range": "bytes=0-1023"}, stream=True)
        return r.status_code in (200, 206)
    except Exception:
        return False


def download_image_as_data_url(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "image/jpeg")
        b64 = base64.b64encode(r.content).decode("utf-8")
        return f"data:{ct};base64,{b64}"
    except Exception:
        return None


def build_prompt(tweet_text, meme_title):
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


def build_prompt_broken(tweet_text, metadata_desc):
    return f"""You are evaluating whether a meme works as a reply to a tweet on social media.

Tweet: "{tweet_text}"

The meme image could not be loaded. Here is a text description of what the meme depicts:
{metadata_desc}

Since the actual image cannot be viewed, you cannot make a reliable judgment.
Respond with ONLY this JSON (no other text):
{{"reply": "broken", "flag": "no"}}"""


def format_metadata_description(meta, meme_title=""):
    parts = []
    title = meta.get("title", meme_title) or meme_title
    if title:
        parts.append(f"Title: {title}")
    img_captions = meta.get("img_captions", [])
    if img_captions:
        caps = "; ".join(img_captions) if isinstance(img_captions, list) else img_captions
        parts.append(f"What the image shows: {caps}")
    meme_captions = meta.get("meme_captions", [])
    if meme_captions:
        caps = "; ".join(meme_captions) if isinstance(meme_captions, list) else meme_captions
        parts.append(f"What the meme means: {caps}")
    metaphors = meta.get("metaphors", [])
    if metaphors and isinstance(metaphors, list):
        met_strs = []
        for m in metaphors:
            if isinstance(m, dict):
                met = m.get("metaphor", "")
                meaning = m.get("meaning", "")
                if met and meaning:
                    met_strs.append(f'"{met}" represents "{meaning}"')
        if met_strs:
            parts.append(f"Visual metaphors: {'; '.join(met_strs)}")
    return "\n".join(parts) if parts else f"Title: {meme_title}"


def parse_response(text):
    text = text.strip()
    try:
        m = re.search(r'\{[^}]+\}', text)
        if m:
            data = json.loads(m.group())
            reply = str(data.get("reply", "no")).lower().strip()
            if reply == "broken":
                return -1
            return 1 if reply in ("yes", "1", "true") else 0
    except (json.JSONDecodeError, AttributeError):
        pass
    lower = text.lower()
    if "broken" in lower:
        return -1
    return 1 if re.search(r'\byes\b', lower) else 0


def call_api(model_id, prompt, image_url=None):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": APP_TITLE,
    }
    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.3,
        "max_tokens": 60,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if r.status_code >= 400 and image_url and attempt == 0:
                data_url = download_image_as_data_url(image_url)
                if data_url:
                    content[-1] = {"type": "image_url", "image_url": {"url": data_url}}
                    payload["messages"][0]["content"] = content
                    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
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


def main():
    if not OPENROUTER_API_KEY:
        print("Error: set OPENROUTER_API_KEY")
        return

    # 1. Load existing CSV, split into keep vs remove
    kept_rows = []
    removed_items = []  # unique items that need new model annotations
    fieldnames = None

    with open(VALIDATION_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["annotator_id"] in REMOVE_IDS:
                key = (row["task_id"], row["candidate_index"])
                # Keep item info for re-annotation
                if not any(r["task_id"] == row["task_id"] and
                          r["candidate_index"] == row["candidate_index"]
                          for r in removed_items):
                    removed_items.append(row)
            else:
                kept_rows.append(row)

    # Build list of ALL unique items from kept rows
    all_items = {}
    for row in kept_rows:
        key = (row["task_id"], row["candidate_index"])
        if key not in all_items:
            all_items[key] = row  # keep item info for re-annotation

    print(f"Keeping {len(kept_rows)} rows (seed-flash + gemini-lite)")
    print(f"Removed annotator IDs: {REMOVE_IDS}")
    print(f"Unique items to annotate: {len(all_items)}")

    # Check which new models already have results (for resumability)
    existing_new = defaultdict(set)
    for row in kept_rows:
        if row["annotator_id"] in {m["annotator_id"] for m in NEW_MODELS}:
            key = (row["task_id"], row["candidate_index"])
            existing_new[key].add(row["annotator_id"])

    # 2. Write back kept rows only
    with open(VALIDATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    # 3. Load MemeCap
    print("Downloading MemeCap metadata...")
    memecap_data = requests.get(MEMECAP_URL, timeout=60).json()
    memecap_lookup = {e.get("post_id", ""): e for e in memecap_data if e.get("post_id")}

    # 4. Run new models on ALL items
    items_list = list(all_items.values())
    image_cache = {}
    total_calls = 0
    for model in NEW_MODELS:
        needed = [item for item in items_list
                  if model["annotator_id"] not in
                  existing_new.get((item["task_id"], item["candidate_index"]), set())]
        total_calls += len(needed)

    print(f"\nTotal API calls needed: {total_calls}")
    print(f"Running new models...\n")

    done = 0
    with open(VALIDATION_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for model in NEW_MODELS:
            model_done = 0
            model_err = 0

            for item in items_list:
                key = (item["task_id"], item["candidate_index"])
                if model["annotator_id"] in existing_new.get(key, set()):
                    continue

                img_url = item["image_url"]
                if img_url not in image_cache:
                    image_cache[img_url] = check_image_accessible(img_url)
                image_ok = image_cache[img_url]

                if image_ok:
                    prompt = build_prompt(item["tweet_text"], item["meme_title"])
                else:
                    meta = memecap_lookup.get(item["meme_post_id"], {})
                    desc = format_metadata_description(meta, item["meme_title"])
                    prompt = build_prompt_broken(item["tweet_text"], desc)

                try:
                    response = call_api(model["id"], prompt, img_url if image_ok else None)
                    pred = parse_response(response)
                except Exception as e:
                    print(f"  [ERR] {model['annotator_id']}: {e}")
                    pred = -1
                    model_err += 1

                new_row = dict(item)
                new_row["annotator_id"] = model["annotator_id"]
                new_row["model_prediction"] = pred
                writer.writerow(new_row)
                f.flush()

                done += 1
                model_done += 1
                if done % 50 == 0:
                    print(f"  [{done}/{total_calls}]")

                time.sleep(SLEEP)

            print(f"  {model['annotator_id']}: {model_done} done, {model_err} errors")

    # 5. Agreement analysis
    print(f"\n{'='*60}")
    print("VALIDATION REPORT (ALL 5 MODELS)")
    print(f"{'='*60}")

    item_preds = defaultdict(dict)
    item_human = {}

    with open(VALIDATION_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["task_id"], row["candidate_index"])
            pred = int(row["model_prediction"])
            item_preds[key][row["annotator_id"]] = pred
            item_human[key] = int(row["human_majority"])

    all_models = sorted({m for preds in item_preds.values() for m in preds})

    complete = {
        k: v for k, v in item_preds.items()
        if len(v) == 5 and all(p != -1 for p in v.values())
    }

    if not complete:
        print("\nNo items with all 5 models completed. Check for errors above.")
        return

    human_yes = sum(1 for k in complete if item_human[k] == 1)
    human_no = len(complete) - human_yes
    print(f"\nItems with all 5 models (no broken): {len(complete)}")
    print(f"Human majority: {human_yes} yes ({human_yes/len(complete)*100:.0f}%), "
          f"{human_no} no ({human_no/len(complete)*100:.0f}%)")

    print(f"\n  {'Model':<30s} {'Acc':>6s} {'Yes%':>6s} {'Prec':>6s} {'Rec':>6s} {'N':>5s}")
    print(f"  {'-'*29} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")

    for mid in all_models:
        tp = fp = tn = fn = 0
        for key, preds in complete.items():
            pred = preds.get(mid)
            if pred is None:
                continue
            human = item_human[key]
            if pred == 1 and human == 1: tp += 1
            elif pred == 1 and human == 0: fp += 1
            elif pred == 0 and human == 0: tn += 1
            elif pred == 0 and human == 1: fn += 1
        total = tp + fp + tn + fn
        if total == 0:
            continue
        acc = (tp + tn) / total * 100
        yes_rate = (tp + fp) / total * 100
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        print(f"  {mid:<30s} {acc:>5.1f}% {yes_rate:>5.1f}% {prec:>5.1f}% {rec:>5.1f}% {total:>5d}")

    # Ensemble
    ens_tp = ens_fp = ens_tn = ens_fn = 0
    for key, preds in complete.items():
        votes = list(preds.values())
        majority = 1 if sum(votes) > len(votes) / 2 else 0
        human = item_human[key]
        if majority == 1 and human == 1: ens_tp += 1
        elif majority == 1 and human == 0: ens_fp += 1
        elif majority == 0 and human == 0: ens_tn += 1
        elif majority == 0 and human == 1: ens_fn += 1
    ens_total = ens_tp + ens_fp + ens_tn + ens_fn
    if ens_total:
        ens_acc = (ens_tp + ens_tn) / ens_total * 100
        ens_yes = (ens_tp + ens_fp) / ens_total * 100
        ens_prec = ens_tp / (ens_tp + ens_fp) * 100 if (ens_tp + ens_fp) > 0 else 0
        ens_rec = ens_tp / (ens_tp + ens_fn) * 100 if (ens_tp + ens_fn) > 0 else 0
        print(f"\n  {'ENSEMBLE (majority vote)':<30s} {ens_acc:>5.1f}% {ens_yes:>5.1f}% {ens_prec:>5.1f}% {ens_rec:>5.1f}% {ens_total:>5d}")

    print(f"\n  Baseline (always no): {human_no/len(complete)*100:.0f}% accuracy")


if __name__ == "__main__":
    main()
