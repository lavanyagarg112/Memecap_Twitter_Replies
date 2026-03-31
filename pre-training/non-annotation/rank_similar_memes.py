#!/usr/bin/env python3
"""
Non-annotation pipeline: Rank MemeCap memes by semantic similarity
to real tweets from HSDSLab/TwitterMemes, powered by a VLM.

Phase 1  VLM describes each tweet + meme image              (parallel, I/O-bound)
Phase 2  Batch-embed descriptions -> candidate pool         (single-thread, fast)
Phase 3  VLM chooses the best 10 from the candidate pool    (parallel, I/O-bound)
Phase 4  Second VLM ranks the chosen 10                     (parallel, I/O-bound)
Phase 5  Write train/val/test CSVs                          (same format as annotation pipeline)

Install:
    pip install datasets requests numpy sentence-transformers python-dotenv Pillow

Usage:
    python rank_similar_memes.py                        # default 4 workers
    python rank_similar_memes.py --workers 6 --limit 1000
    python rank_similar_memes.py --dry-run              # cost estimate only
"""

import argparse
import base64
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ========================= CONFIG =========================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_REFERER = "http://localhost"
APP_TITLE = "twitter-meme-ranker"

# Cheapest vision-capable model on OpenRouter (Phase 1: descriptions)
VLM_MODEL = {
    "id": "bytedance-seed/seed-1.6-flash",
    "cost_per_1m_input": 0.075,
    "cost_per_1m_output": 0.30,
}

# Same fast model for Phase 3 candidate selection
SELECT_MODEL = VLM_MODEL

# Stronger model for Phase 4 final ranking
RERANK_MODEL = {
    "id": "google/gemini-3-flash-preview",
    "cost_per_1m_input": 0.50,
    "cost_per_1m_output": 3.00,
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MEMECAP_URL = "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-trainval.json"
FLAGGED_FILE = Path(__file__).resolve().parent.parent / "flagged_memes.json"

TOP_K = 10
DEFAULT_CANDIDATE_POOL = 20  # embedding shortlist sent to the VLM chooser

# Conservative default — another parallel job may be running on this machine.
# Increase if machine is free (e.g. --workers 12).
DEFAULT_WORKERS = 4
DEFAULT_BUDGET_USD = 10.0
MAX_RETRIES = 2

# Token estimates for cost calculation (image tokens counted separately by API)
EST_INPUT_TOKENS_DESCRIBE = 1800
EST_OUTPUT_TOKENS_DESCRIBE = 200
EST_INPUT_TOKENS_SELECT = 7000
EST_OUTPUT_TOKENS_SELECT = 120
EST_INPUT_TOKENS_RERANK = 5000
EST_OUTPUT_TOKENS_RERANK = 180

OUTPUT_FILE = "twitter_meme_rankings.jsonl"
DESCRIPTIONS_CACHE = ".descriptions_cache.jsonl"

# Train/val/test split (same ratios as annotation pipeline)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

# Output CSV fields
OUTPUT_CSV_FIELDS = [
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
    "similarity_score",
]


# ========================= HELPERS =========================

def pil_to_data_url(img) -> str:
    """Convert PIL Image to base64 data URL for the API."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def check_url(url: str) -> bool:
    """Quick check if an image URL is reachable."""
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


def get_image_url(row: dict) -> Optional[str]:
    """Get a usable image URL — try img_link first, fall back to base64."""
    img_link = row.get("img_link", "")
    if img_link and check_url(img_link):
        return img_link
    img = row.get("image")
    if img is not None:
        try:
            return pil_to_data_url(img)
        except Exception:
            pass
    return None


def call_openrouter(prompt: str, image_url: Optional[str] = None,
                    max_tokens: int = 250, model: Optional[Dict] = None) -> str:
    """Send a request to OpenRouter. Includes image if provided."""
    if image_url:
        content: Any = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        content = prompt
    return call_openrouter_content(content, max_tokens=max_tokens, model=model)


def call_openrouter_content(content: Any, max_tokens: int = 250,
                            model: Optional[Dict] = None) -> str:
    """Send a raw multimodal content payload to OpenRouter."""
    if model is None:
        model = VLM_MODEL
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": APP_TITLE,
    }

    payload = {
        "model": model["id"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers,
                              json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            c = data["choices"][0]["message"]["content"]
            if not c:
                raise ValueError("Empty response")
            return c.strip()
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def description_cost_per_tweet() -> float:
    return (
        (EST_INPUT_TOKENS_DESCRIBE / 1_000_000) * VLM_MODEL["cost_per_1m_input"]
        + (EST_OUTPUT_TOKENS_DESCRIBE / 1_000_000) * VLM_MODEL["cost_per_1m_output"]
    )


def selection_cost_per_tweet() -> float:
    return (
        (EST_INPUT_TOKENS_SELECT / 1_000_000) * SELECT_MODEL["cost_per_1m_input"]
        + (EST_OUTPUT_TOKENS_SELECT / 1_000_000) * SELECT_MODEL["cost_per_1m_output"]
    )


def ranking_cost_per_tweet() -> float:
    return (
        (EST_INPUT_TOKENS_RERANK / 1_000_000) * RERANK_MODEL["cost_per_1m_input"]
        + (EST_OUTPUT_TOKENS_RERANK / 1_000_000) * RERANK_MODEL["cost_per_1m_output"]
    )


def row_cost(tweet_id: str, cached_ids: set) -> float:
    cost = selection_cost_per_tweet() + ranking_cost_per_tweet()
    if tweet_id not in cached_ids:
        cost += description_cost_per_tweet()
    return cost


def estimate_cost(rows: List[Dict[str, Any]], cached_ids: set) -> float:
    """Estimate total API cost in USD for the rows in scope."""
    total = 0.0
    for i, row in enumerate(rows):
        tid = row.get("id", str(i))
        total += row_cost(tid, cached_ids)
    return total


def fit_rows_to_budget(rows: List[Dict[str, Any]], cached_ids: set,
                       budget_usd: float) -> Tuple[int, float]:
    """Return how many rows from the front fit within the budget."""
    total = 0.0
    keep = 0
    for i, row in enumerate(rows):
        tid = row.get("id", str(i))
        cost = row_cost(tid, cached_ids)
        if keep > 0 and total + cost > budget_usd:
            break
        if keep == 0 and cost > budget_usd:
            return 0, 0.0
        total += cost
        keep += 1
    return keep, total


# ========================= DATA LOADING =========================

def load_excluded() -> set:
    """Load excluded post_ids from flagged_memes.json."""
    if not FLAGGED_FILE.exists():
        print(f"  Warning: {FLAGGED_FILE} not found — no memes excluded.")
        print(f"  Run 'python flag_memes.py' in pre-training/ to review flagged content.")
        return set()
    with open(FLAGGED_FILE, encoding="utf-8") as f:
        data = json.load(f)
    excluded = set(data.get("excluded_post_ids", []))
    print(f"  Loaded blocklist: {len(excluded)} memes excluded.")
    return excluded


def fetch_memecap() -> List[Dict[str, Any]]:
    """Download MemeCap dataset from GitHub, excluding flagged memes."""
    print(f"Downloading MemeCap from {MEMECAP_URL} ...")
    r = requests.get(MEMECAP_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    print(f"  Loaded {len(data)} memes.")

    excluded = load_excluded()
    if excluded:
        before = len(data)
        data = [m for m in data if m.get("post_id", "") not in excluded]
        print(f"  Filtered: {before} -> {len(data)} memes ({before - len(data)} removed).")

    return data


def to_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return "" if np.isnan(val) else str(val)
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def build_meme_text(meme: Dict[str, Any]) -> str:
    """Build rich text representation of a MemeCap meme for embedding."""
    parts = []
    title = to_str(meme.get("title", ""))
    if title:
        parts.append(title)
    for cap in meme.get("img_captions", []):
        s = to_str(cap)
        if s:
            parts.append(s)
    for cap in meme.get("meme_captions", []):
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


def build_memecap_embeddings(
    memes: List[Dict[str, Any]], model: SentenceTransformer,
) -> np.ndarray:
    """Pre-compute normalized embeddings for all MemeCap memes."""
    texts = [build_meme_text(m) for m in memes]
    print(f"  Encoding {len(texts)} MemeCap memes ...")
    embs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embs)


_KEEP_FIELDS = {"id", "post_text", "ocr", "img_link"}


def load_twitter_memes(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load TwitterMemes from HuggingFace using streaming to avoid disk usage.

    The full dataset is ~11GB (images embedded in parquet). Streaming fetches
    rows on demand without caching to disk.

    Only text fields + img_link are kept; the heavy PIL 'image' is discarded
    immediately to avoid OOM on large runs.  For uncached tweets that still
    need Phase 1 descriptions, get_image_url() will fetch the image on
    demand from img_link.
    """
    from datasets import load_dataset

    print("Loading HSDSLab/TwitterMemes from HuggingFace (streaming) ...")
    ds = load_dataset("HSDSLab/TwitterMemes", split="train", streaming=True)

    rows = []
    iterator = iter(ds)
    try:
        for i, row in enumerate(iterator):
            if limit and i >= limit:
                break
            # Keep only lightweight text fields — drop PIL image immediately
            slim = {k: row[k] for k in _KEEP_FIELDS if k in row}
            rows.append(slim)
            if (i + 1) % 500 == 0:
                print(f"    streamed {i + 1} tweets ({len(rows)} kept) ...")
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    print(f"  Loaded {len(rows)} tweets (text-only).")
    return rows


# ========================= PHASE 1: VLM DESCRIPTIONS =========================

DESCRIBE_PROMPT = """Analyze this tweet and its attached meme image. Write a concise semantic description covering:
1. Main topic of the tweet
2. The humor, irony, or emotional tone of the meme
3. Cultural references or internet meme templates used
4. How the meme relates to the tweet

Tweet: "{post_text}"
{ocr_line}
Write a single paragraph (3-5 sentences) capturing the semantic meaning."""


def describe_one(row: dict) -> Optional[str]:
    """Use VLM to semantically describe a tweet + its meme image."""
    img_url = get_image_url(row)
    if not img_url:
        return None

    ocr = to_str(row.get("ocr", ""))
    ocr_line = f'OCR text from image: "{ocr}"' if ocr else ""
    prompt = DESCRIBE_PROMPT.format(
        post_text=row.get("post_text", ""),
        ocr_line=ocr_line,
    )
    try:
        return call_openrouter(prompt, img_url, max_tokens=250)
    except Exception:
        return None


def load_description_cache(cache_path: Path) -> Dict[str, str]:
    """Load cached VLM descriptions (for resumability)."""
    cache = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["tweet_id"]] = rec["description"]
    return cache


def run_phase1(
    twitter_ds, cache_path: Path, workers: int,
) -> Dict[str, str]:
    """Phase 1: Generate VLM descriptions in parallel. Resumable."""
    cache = load_description_cache(cache_path)
    if cache:
        print(f"  Resuming Phase 1: {len(cache)} descriptions cached.")

    # Figure out which tweets still need descriptions
    todo = []
    for i in range(len(twitter_ds)):
        tid = twitter_ds[i].get("id", str(i))
        if tid not in cache:
            todo.append(i)

    if not todo:
        print("  Phase 1: all descriptions cached — skipping.")
        return cache

    print(f"  Phase 1: generating descriptions for {len(todo)} tweets "
          f"({workers} workers) ...\n")

    write_lock = Lock()
    done = [0]
    errors = [0]

    def worker(idx: int) -> Tuple[str, Optional[str]]:
        row = twitter_ds[idx]
        tid = row.get("id", str(idx))
        desc = describe_one(row)
        return tid, desc

    with open(cache_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, idx): idx for idx in todo}

            for future in as_completed(futures):
                try:
                    tid, desc = future.result()
                    if desc:
                        with write_lock:
                            cache[tid] = desc
                            fout.write(json.dumps(
                                {"tweet_id": tid, "description": desc},
                                ensure_ascii=False,
                            ) + "\n")
                            fout.flush()
                            done[0] += 1
                    else:
                        errors[0] += 1
                except Exception:
                    errors[0] += 1

                total = done[0] + errors[0]
                if total % 100 == 0 or total == len(todo):
                    print(f"    Phase 1: [{total}/{len(todo)}] "
                          f"done={done[0]}  errors={errors[0]}")

    print(f"  Phase 1 complete: {done[0]} new descriptions, "
          f"{errors[0]} errors, {len(cache)} total cached.\n")
    return cache


# ========================= PHASE 2: EMBEDDING SIMILARITY =========================

def run_phase2(
    twitter_ds,
    descriptions: Dict[str, str],
    embed_model: SentenceTransformer,
    memecap_embeddings: np.ndarray,
    memecap_data: List[Dict],
    top_k: int,
) -> List[dict]:
    """Phase 2: Batch-embed descriptions and compute top-K similar memes."""
    print(f"  Phase 2: building embedding candidate pools ...")

    # Build aligned arrays
    tweet_ids = []
    query_texts = []
    tweet_texts = {}
    tweet_img_links = {}

    for i in range(len(twitter_ds)):
        tid = twitter_ds[i].get("id", str(i))
        if tid in descriptions:
            tweet_ids.append(tid)
            # Combine tweet text + VLM description for richer embedding
            post_text = twitter_ds[i].get("post_text", "")
            tweet_texts[tid] = post_text
            tweet_img_links[tid] = twitter_ds[i].get("img_link", "")
            query_texts.append(f"{post_text} {descriptions[tid]}")

    if not query_texts:
        print("  Phase 2: nothing to embed.")
        return []

    query_embeddings = embed_model.encode(
        query_texts, show_progress_bar=True, normalize_embeddings=True,
    )
    query_embeddings = np.array(query_embeddings)

    # Cosine similarity: (N_tweets, N_memecap)
    print(f"  Computing similarity matrix ({len(query_texts)} x {len(memecap_data)}) ...")
    sim_matrix = query_embeddings @ memecap_embeddings.T

    results = []
    for i, tid in enumerate(tweet_ids):
        scores = sim_matrix[i]
        top_indices = np.argsort(scores)[::-1][:top_k]

        ranked_memes = []
        for rank, idx in enumerate(top_indices):
            meme = memecap_data[idx]
            ranked_memes.append({
                "rank": rank + 1,
                "embedding_rank": rank + 1,
                "memecap_idx": int(idx),
                "memecap_post_id": meme.get("post_id", ""),
                "title": meme.get("title", ""),
                "image_url": meme.get("url", ""),
                "similarity_score": round(float(scores[idx]), 4),
                "img_captions": meme.get("img_captions", []),
                "meme_captions": meme.get("meme_captions", []),
                "metaphors": meme.get("metaphors", []),
            })

        results.append({
            "tweet_id": tid,
            "post_text": tweet_texts.get(tid, ""),
            "img_link": tweet_img_links.get(tid, ""),
            "tweet_description": descriptions[tid],
            "top_memes": ranked_memes,
            "selection_stage_method": "embedding_pool",
        })

    print(f"  Phase 2 complete: {len(results)} tweets ranked.\n")
    return results


# ========================= PHASE 3: VLM SELECTION =========================

SELECT_PROMPT = """You are screening meme candidates for semantic similarity to an original tweet and meme image.

Original tweet: "{post_text}"
Original meme description: "{description}"

Candidate memes (numbered):
{candidates_text}

Select the {top_k} candidates that are the strongest semantic matches.
Consider topic, humor style, emotional tone, cultural references, and visual template.

Return ONLY a JSON array of exactly {top_k} candidate numbers.
Example: [3, 7, 1, 5, 9, 2, 8, 4, 10, 6]"""


RERANK_PROMPT = """You are ranking meme candidates by semantic similarity to an original tweet and meme image.

Original tweet: "{post_text}"
Original meme description: "{description}"

Candidate memes (numbered):
{candidates_text}

Rank all {top_k} candidates from most similar to least similar.
Consider topic, humor style, emotional tone, cultural references, and visual template.

Return ONLY a JSON array of all {top_k} candidate numbers, most similar first.
Example: [3, 7, 1, 5, 9, 2, 8, 4, 10, 6]"""


def truncate_text(text: str, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."


def build_candidate_text(meme: Dict[str, Any], display_idx: int,
                         include_score: bool = False) -> str:
    lines = [f"{display_idx}. {to_str(meme.get('title', '')) or '(untitled meme)'}"]
    if include_score:
        score = meme.get("similarity_score")
        if score is not None:
            lines.append(f"Embedding similarity: {score}")
    summary = truncate_text(build_meme_text(meme))
    if summary:
        lines.append(f"Summary: {summary}")
    return "\n".join(lines)


def build_candidate_blocks(prompt: str, original_img_url: Optional[str],
                           candidates: List[Dict[str, Any]],
                           include_score: bool = False) -> List[Dict[str, Any]]:
    blocks = [{"type": "text", "text": prompt}]
    if original_img_url:
        blocks.append({"type": "text", "text": "Original meme image:"})
        blocks.append({"type": "image_url", "image_url": {"url": original_img_url}})

    for i, meme in enumerate(candidates, start=1):
        blocks.append({
            "type": "text",
            "text": build_candidate_text(meme, i, include_score=include_score),
        })
        image_url = to_str(meme.get("image_url", ""))
        if image_url:
            blocks.append({"type": "image_url", "image_url": {"url": image_url}})
    return blocks


def parse_number_array(response: str) -> List[int]:
    match = re.search(r"\[[^\]]*\]", response, flags=re.S)
    if not match:
        return []
    try:
        arr = json.loads(match.group())
    except Exception:
        return []
    if not isinstance(arr, list):
        return []

    nums = []
    for item in arr:
        try:
            nums.append(int(item))
        except Exception:
            continue
    return nums


def normalize_candidate_positions(nums: List[int], n_candidates: int,
                                  keep: int) -> List[int]:
    result = []
    seen = set()
    for n in nums:
        pos = n - 1
        if 0 <= pos < n_candidates and pos not in seen:
            result.append(pos)
            seen.add(pos)
        if len(result) >= keep:
            return result[:keep]

    for pos in range(n_candidates):
        if pos not in seen:
            result.append(pos)
        if len(result) >= keep:
            break
    return result[:keep]


def select_one(
    tweet_id: str,
    post_text: str,
    description: str,
    candidates: List[Dict[str, Any]],
    img_url: Optional[str],
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Use a VLM to choose the best 10 candidates from the embedding pool."""
    if len(candidates) <= TOP_K:
        selected = [dict(meme) for meme in candidates[:TOP_K]]
        return tweet_id, selected, "embedding_topk"

    lines = [build_candidate_text(meme, i + 1, include_score=True)
             for i, meme in enumerate(candidates)]
    prompt = SELECT_PROMPT.format(
        post_text=post_text,
        description=description,
        candidates_text="\n\n".join(lines),
        top_k=TOP_K,
    )
    blocks = build_candidate_blocks(prompt, img_url, candidates, include_score=True)

    method = "vlm_select"
    try:
        response = call_openrouter_content(blocks, max_tokens=120, model=SELECT_MODEL)
        positions = normalize_candidate_positions(
            parse_number_array(response),
            len(candidates),
            TOP_K,
        )
    except Exception:
        positions = list(range(TOP_K))
        method = "embedding_topk"

    selected = []
    for order, pos in enumerate(positions, start=1):
        meme = dict(candidates[pos])
        meme["selection_rank"] = order
        selected.append(meme)
    return tweet_id, selected, method


def run_phase3(
    twitter_ds,
    phase2_results: List[dict],
    workers: int,
) -> List[dict]:
    """Phase 3: choose the best 10 candidates from the embedding pool."""
    print(f"  Phase 3: VLM selecting top-{TOP_K} from each candidate pool "
          f"({workers} workers) ...")

    tid_to_row = {}
    for i in range(len(twitter_ds)):
        tid_to_row[twitter_ds[i].get("id", str(i))] = twitter_ds[i]

    selected = {}
    methods = {}
    done = [0]
    write_lock = Lock()

    def worker(rec: Dict[str, Any]):
        row = tid_to_row.get(rec["tweet_id"], {})
        img_url = to_str(row.get("img_link", "")) or None
        return select_one(
            rec["tweet_id"],
            rec["post_text"],
            rec["tweet_description"],
            rec["top_memes"],
            img_url,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, rec): rec["tweet_id"] for rec in phase2_results}
        for future in as_completed(futures):
            tid, chosen, method = future.result()
            with write_lock:
                selected[tid] = chosen
                methods[tid] = method
                done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(phase2_results):
                print(f"    Phase 3: [{done[0]}/{len(phase2_results)}]")

    final_results = []
    for rec in phase2_results:
        tid = rec["tweet_id"]
        new_rec = dict(rec)
        new_rec["candidate_pool"] = [dict(meme) for meme in rec["top_memes"]]
        new_rec["top_memes"] = selected.get(tid, [dict(m) for m in rec["top_memes"][:TOP_K]])
        new_rec["selection_stage_method"] = methods.get(tid, "embedding_topk")
        final_results.append(new_rec)

    print(f"  Phase 3 complete: {len(final_results)} tweets selected.\n")
    return final_results


# ========================= PHASE 4: SECOND-VLM RANKING =========================

def rerank_one(
    tweet_id: str,
    post_text: str,
    description: str,
    candidates: List[Dict[str, Any]],
    img_url: Optional[str],
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Use a second VLM to rank the 10 chosen candidates."""
    lines = [build_candidate_text(meme, i + 1) for i, meme in enumerate(candidates)]
    candidates_text = "\n\n".join(lines)

    prompt = RERANK_PROMPT.format(
        post_text=post_text,
        description=description,
        candidates_text=candidates_text,
        top_k=len(candidates),
    )
    blocks = build_candidate_blocks(prompt, img_url, candidates, include_score=False)

    method = "vlm_rank"
    try:
        response = call_openrouter_content(blocks, max_tokens=150, model=RERANK_MODEL)
        positions = normalize_candidate_positions(
            parse_number_array(response),
            len(candidates),
            len(candidates),
        )
    except Exception:
        positions = list(range(len(candidates)))
        method = "selection_order"

    ranked = []
    for rank, pos in enumerate(positions, start=1):
        meme = dict(candidates[pos])
        meme["rank"] = rank
        ranked.append(meme)
    return tweet_id, ranked, method


def run_phase4(
    twitter_ds,
    phase2_results: List[dict],
    workers: int,
) -> List[dict]:
    """Phase 4: second VLM ranks the 10 chosen candidates."""
    print(f"  Phase 4: second-VLM ranking for {len(phase2_results)} tweets "
          f"({workers} workers) ...")
    tid_to_row = {}
    for i in range(len(twitter_ds)):
        tid_to_row[twitter_ds[i].get("id", str(i))] = twitter_ds[i]

    write_lock = Lock()
    done = [0]
    reranked = {}
    methods = {}

    def worker(rec: Dict[str, Any]):
        row = tid_to_row.get(rec["tweet_id"], {})
        img_url = to_str(row.get("img_link", "")) or None
        return rerank_one(
            rec["tweet_id"],
            rec["post_text"],
            rec["tweet_description"],
            rec["top_memes"],
            img_url,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, rec): rec["tweet_id"] for rec in phase2_results}
        for future in as_completed(futures):
            tid, ranked, method = future.result()
            with write_lock:
                reranked[tid] = ranked
                methods[tid] = method
                done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(phase2_results):
                print(f"    Phase 4: [{done[0]}/{len(phase2_results)}]")

    final_results = []
    for rec in phase2_results:
        tid = rec["tweet_id"]
        new_rec = dict(rec)
        new_rec["selected_memes"] = [dict(meme) for meme in rec["top_memes"]]
        new_rec["top_memes"] = reranked.get(tid, [dict(m) for m in rec["top_memes"]])
        new_rec["ranking_stage_method"] = methods.get(tid, "selection_order")
        new_rec["selection_method"] = (
            f"{new_rec.get('selection_stage_method', 'unknown')}"
            f"+{new_rec['ranking_stage_method']}"
        )
        final_results.append(new_rec)

    print(f"  Phase 4 complete: {len(final_results)} tweets ranked.\n")
    return final_results


# ========================= PHASE 5: TRAIN/VAL/TEST CSVs =========================

def format_captions(captions: Any) -> str:
    """Join list of captions into pipe-separated string (matches annotation pipeline)."""
    if not captions:
        return ""
    if isinstance(captions, list):
        return " | ".join(str(c).strip() for c in captions if str(c).strip())
    return str(captions).strip()


def format_metaphors(metaphors: Any) -> str:
    """Format metaphor list into readable string (matches annotation pipeline)."""
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


def run_phase5(results: List[dict], script_dir: Path):
    """Phase 5: Convert ranked results to train/val/test CSVs
    in the same format as the annotation pipeline."""
    print("  Phase 5: generating train/val/test CSVs ...")

    # Build flat rows grouped by task_id (= tweet_id)
    task_rows: Dict[str, List[dict]] = {}
    for rec in results:
        tid = rec["tweet_id"]
        rows = []
        for meme in rec["top_memes"]:
            rows.append({
                "task_id": tid,
                "tweet_text": rec["post_text"],
                "meme_post_id": meme.get("memecap_post_id", ""),
                "image_url": meme.get("image_url", ""),
                "meme_title": meme.get("title", ""),
                "img_captions": format_captions(meme.get("img_captions")),
                "meme_captions": format_captions(meme.get("meme_captions")),
                "metaphors": format_metaphors(meme.get("metaphors")),
                "selection_method": rec.get("selection_method", "vlm_select+vlm_rank"),
                "candidate_index": meme["rank"] - 1,
                "rank": meme["rank"],
                "similarity_score": meme.get("similarity_score", 0.0),
            })
        task_rows[tid] = rows

    # Split by task
    task_ids = sorted(task_rows.keys())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(task_ids)

    n = len(task_ids)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": task_ids[:train_end],
        "val": task_ids[train_end:val_end],
        "test": task_ids[val_end:],
    }

    for split_name, split_tasks in splits.items():
        out_path = script_dir / f"{split_name}.csv"
        rows = []
        for tid in split_tasks:
            rows.extend(task_rows.get(tid, []))

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        n_tasks = len(split_tasks)
        print(f"    {out_path.name:12s}: {n_tasks:6d} tasks, {len(rows):7d} rows")

    print(f"  Phase 5 complete.\n")


# ========================= MAIN =========================

def main():
    parser = argparse.ArgumentParser(
        description="Rank MemeCap memes by similarity to TwitterMemes "
                    "tweets using an embedding pool plus two VLM passes.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel workers (default {DEFAULT_WORKERS}; kept low for "
             f"shared machines — increase with --workers 12 if machine is free)",
    )
    parser.add_argument(
        "--budget", type=float, default=DEFAULT_BUDGET_USD,
        help=f"Max API spend in USD (default ${DEFAULT_BUDGET_USD:.2f})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max tweets to process (default: all 174k)",
    )
    parser.add_argument(
        "--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL,
        help=f"Embedding shortlist size before VLM selection (default {DEFAULT_CANDIDATE_POOL})",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Deprecated flag; VLM selection and ranking now always run.",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE,
        help=f"Output JSONL path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate cost and exit without calling API",
    )
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY in .env or environment.")
        sys.exit(1)

    if args.candidate_pool < TOP_K:
        print(f"ERROR: --candidate-pool must be at least {TOP_K}.")
        sys.exit(1)

    # ---- Load cache and tweets first so budgeting is scoped correctly ----
    script_dir = Path(__file__).resolve().parent
    cache_path = script_dir / DESCRIPTIONS_CACHE
    output_path = script_dir / args.output

    cached = load_description_cache(cache_path)
    cached_ids = set(cached.keys())

    twitter_ds = load_twitter_memes(args.limit)
    n_tweets = len(twitter_ds)
    n_cached = sum(
        1
        for i, row in enumerate(twitter_ds)
        if row.get("id", str(i)) in cached_ids
    )
    n_remaining = n_tweets - n_cached

    # ---- Cost estimate ----
    est_cost = estimate_cost(twitter_ds, cached_ids)
    print(f"\n{'=' * 55}")
    print(f"  VLM model:   {VLM_MODEL['id']}")
    print(f"  Select:      {SELECT_MODEL['id']}")
    print(f"  Rank:        {RERANK_MODEL['id']}")
    print(f"  Tweets:      {n_remaining} remaining / {n_tweets} total "
          f"({n_cached} cached)")
    print(f"  Workers:     {args.workers}")
    print(f"  Candidate pool: {args.candidate_pool} -> {TOP_K}")
    print(f"  Est. cost:   ${est_cost:.2f}")
    print(f"  Budget:      ${args.budget:.2f}")
    print(f"  Output:      {output_path}")
    print(f"{'=' * 55}\n")

    # Trim to budget
    budget_tweets = n_tweets
    trimmed_cost = est_cost
    if n_tweets > 0 and est_cost > args.budget:
        budget_tweets, trimmed_cost = fit_rows_to_budget(twitter_ds, cached_ids, args.budget)
        print(f"  Budget cap: trimming to {budget_tweets} tweets "
              f"(~${trimmed_cost:.2f}).\n")

    if budget_tweets < n_tweets:
        twitter_ds = twitter_ds[:budget_tweets]
        n_tweets = len(twitter_ds)
        n_cached = sum(
            1
            for i, row in enumerate(twitter_ds)
            if row.get("id", str(i)) in cached_ids
        )
        n_remaining = n_tweets - n_cached

    if args.dry_run:
        print("Dry run complete — exiting.")
        return

    if n_tweets == 0:
        print("Nothing to process.")
        return

    # ---- Load datasets and embeddings ----
    memecap = fetch_memecap()

    print(f"\nLoading embedding model ({EMBEDDING_MODEL}) ...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    memecap_embeddings = build_memecap_embeddings(memecap, embed_model)

    # ---- Phase 1: VLM descriptions ----
    descriptions = run_phase1(twitter_ds, cache_path, args.workers)

    if not descriptions:
        print("No descriptions generated — exiting.")
        return

    # ---- Phase 2: Embedding candidate pool ----
    results = run_phase2(
        twitter_ds, descriptions, embed_model,
        memecap_embeddings, memecap, args.candidate_pool,
    )

    # ---- Phase 3: VLM selection ----
    if results:
        results = run_phase3(twitter_ds, results, args.workers)

    # ---- Phase 4: second-VLM ranking ----
    if results:
        results = run_phase4(twitter_ds, results, args.workers)

    # ---- Save JSONL (intermediate / inspection) ----
    for rec in results:
        rec["top_memes"] = rec["top_memes"][:TOP_K]

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- Phase 5: Train/val/test CSVs ----
    run_phase5(results, script_dir)

    print(f"{'=' * 55}")
    print(f"  DONE: {len(results)} tweets with top-{TOP_K} meme rankings")
    print(f"  JSONL:  {output_path}")
    print(f"  CSVs:   {script_dir}/train.csv, val.csv, test.csv")
    print(f"  Cache:  {cache_path}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
