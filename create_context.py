"""
Generate synthetic Twitter/X posts from MemeCap using OpenRouter,
with everything pulled online.

No local images required.
No command-line args required.

Install:
    pip install requests

Run:
    python generate_memecap_online.py
"""

import json
import os
import re
import time
import base64
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
load_dotenv()


# =========================
# CONFIG
# =========================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Use any vision-capable model available on OpenRouter
OPENROUTER_MODEL = "qwen/qwen2.5-vl-72b-instruct"

# Pick one:
DATASET_SPLIT = "test"   # "test" or "trainval"

# Source dataset files from the GitHub repo
DATASET_URLS = {
    "test": "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-test.json",
    "trainval": "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data/memes-trainval.json",
}

OUTPUT_JSONL_PATH = "memecap_synthetic_twitter_posts.jsonl"

START_INDEX = 0
LIMIT = None          # set to None for all
SLEEP_SECONDS = 1.0

TEMPERATURE = 0.9
MAX_TOKENS = 120

# If direct image URL fails with your chosen model/provider,
# this fallback downloads the image and sends it as base64 data URL instead.
USE_BASE64_IMAGE_FALLBACK = True

# Prompt controls
USE_TITLE = True
USE_IMAGE_CAPTIONS = True
USE_METAPHORS = True
USE_REFERENCE_MEME_CAPTIONS = False
USE_POST_ID = False

# OpenRouter optional metadata headers
HTTP_REFERER = "http://localhost"
APP_TITLE = "memecap-online-twitter-generator"


# =========================
# CONSTANTS
# =========================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# =========================
# HELPERS
# =========================

def fetch_json(url: str) -> Any:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        return "\n".join(safe_str(v) for v in x if safe_str(v))
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^['\"]|['\"]$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))


def metaphors_to_text(metaphors: Any) -> str:
    """
    MemeCap rows store metaphors like:
    [
      {"metaphor": "A woman", "meaning": "Meme poster"},
      ...
    ]
    """
    if not metaphors:
        return ""

    if isinstance(metaphors, list):
        lines = []
        for item in metaphors:
            if isinstance(item, dict):
                m = safe_str(item.get("metaphor"))
                meaning = safe_str(item.get("meaning"))
                if m or meaning:
                    if meaning:
                        lines.append(f"{m} -> {meaning}")
                    else:
                        lines.append(m)
            else:
                lines.append(safe_str(item))
        return "\n".join(line for line in lines if line)

    return safe_str(metaphors)


def build_prompt(example: Dict[str, Any]) -> str:
    title = safe_str(example.get("title"))
    img_captions = safe_str(example.get("img_captions"))
    meme_captions = safe_str(example.get("meme_captions"))
    metaphors = metaphors_to_text(example.get("metaphors"))
    post_id = safe_str(example.get("post_id"))

    parts = [
        "You are given a meme and metadata describing it.",
        "",
        "Generate exactly ONE realistic Twitter/X post such that:",
        "1. the post could plausibly appear on social media",
        "2. the meme would make sense as a reply to it",
        "3. the relationship should feel natural, funny, sarcastic, relatable, or mocking",
        "",
        "Rules:",
        "- Output only the original tweet/post",
        "- Do not output the meme reply",
        "- Do not describe the image",
        "- Do not mention 'this meme' or 'reaction image'",
        "- Do not write as if you are uploading the meme",
        "- Make the post specific enough that the meme reply feels justified",
        "- Keep it short and realistic",
        "- Avoid slurs, threats, and explicit sexual content",
        "",
        "Meme metadata:"
    ]

    if USE_TITLE:
        parts.append(f"Title: {title or '[missing]'}")

    if USE_IMAGE_CAPTIONS:
        parts.append(f"Literal image captions: {img_captions or '[missing]'}")

    if USE_METAPHORS:
        parts.append(f"Visual metaphors: {metaphors or '[missing]'}")

    if USE_REFERENCE_MEME_CAPTIONS:
        parts.append(f"Reference meme captions: {meme_captions or '[missing]'}")

    if USE_POST_ID:
        parts.append(f"Post ID: {post_id or '[missing]'}")

    parts.extend([
        "",
        "Generate the original tweet/post now."
    ])

    return "\n".join(parts)


def download_image_as_data_url(image_url: str) -> Optional[str]:
    try:
        r = requests.get(image_url, timeout=30)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "image/jpeg")
        b64 = base64.b64encode(r.content).decode("utf-8")
        return f"data:{content_type};base64,{b64}"
    except Exception:
        return None


def build_content(prompt: str, image_url: Optional[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    if image_url and looks_like_url(image_url):
        content.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    return content


def call_openrouter(prompt: str, image_url: Optional[str]) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": APP_TITLE,
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": build_content(prompt, image_url)
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)

    # Optional fallback if direct image URL fails
    if r.status_code >= 400 and USE_BASE64_IMAGE_FALLBACK and image_url:
        data_url = download_image_as_data_url(image_url)
        if data_url:
            payload["messages"][0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)

    r.raise_for_status()
    data = r.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenRouter response: {json.dumps(data)[:1200]}") from e


def main():
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_KEY_HERE":
        raise ValueError("Set OPENROUTER_API_KEY at the top of the script.")

    if DATASET_SPLIT not in DATASET_URLS:
        raise ValueError(f"DATASET_SPLIT must be one of: {list(DATASET_URLS.keys())}")

    dataset_url = DATASET_URLS[DATASET_SPLIT]
    rows = fetch_json(dataset_url)

    if not isinstance(rows, list):
        raise ValueError("Expected dataset JSON to be a list of meme examples.")

    end = len(rows) if LIMIT is None else min(len(rows), START_INDEX + LIMIT)

    print(f"Loaded {len(rows)} rows from {dataset_url}")
    print(f"Processing rows [{START_INDEX}:{end}]")
    print(f"Writing to {OUTPUT_JSONL_PATH}")

    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as out:
        for idx in range(START_INDEX, end):
            example = rows[idx]
            image_url = safe_str(example.get("url"))
            prompt = build_prompt(example)

            try:
                generated = call_openrouter(prompt, image_url if looks_like_url(image_url) else None)
                generated = clean_output(generated)

                record = {
                    "index": idx,
                    "post_id": safe_str(example.get("post_id")),
                    "title": safe_str(example.get("title")),
                    "image_url": image_url,
                    "synthetic_twitter_post": generated,
                    "source_example": example,
                }
                print(f"[OK] {idx}: {generated}")

            except Exception as e:
                record = {
                    "index": idx,
                    "post_id": safe_str(example.get("post_id")),
                    "title": safe_str(example.get("title")),
                    "image_url": image_url,
                    "error": str(e),
                    "source_example": example,
                }
                print(f"[ERR] {idx}: {e}")

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(SLEEP_SECONDS)

    print("Done.")


if __name__ == "__main__":
    main()
