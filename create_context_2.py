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

OUTPUT_JSONL_PATH_1 = "memecap_reply_source_tweet_1.jsonl"
OUTPUT_JSONL_PATH_2 = "memecap_reply_source_tweet_2.jsonl"

GENERATE_TWO_CANDIDATES = True

TEMPERATURE_CANDIDATE_1 = 0.65
TEMPERATURE_CANDIDATE_2 = 0.9

START_INDEX = 0
LIMIT = None
SLEEP_SECONDS = 1.0

MAX_TOKENS = 60

USE_BASE64_IMAGE_FALLBACK = True

USE_TITLE = True
USE_IMAGE_CAPTIONS = True
USE_METAPHORS = True
USE_REFERENCE_MEME_CAPTIONS = False
USE_POST_ID = False

HTTP_REFERER = "http://localhost"
APP_TITLE = "memecap-reply-source-generator"


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


def build_prompt(example: Dict[str, Any], candidate_num: int = 1) -> str:
    title = safe_str(example.get("title"))
    img_captions = safe_str(example.get("img_captions"))
    meme_captions = safe_str(example.get("meme_captions"))
    metaphors = metaphors_to_text(example.get("metaphors"))
    post_id = safe_str(example.get("post_id"))

    parts = [
        "You are given a meme and metadata describing it.",
        "",
        f"Generate exactly ONE realistic Twitter/X post that could naturally appear BEFORE the meme, where the meme would make sense as a reply.",
        "",
        "Your output must be the original tweet only.",
        "",
        "What the output should be like:",
        "- a normal standalone tweet that a real person might post",
        "- natural, conversational, and plausible on its own",
        "- something that invites a reaction, eye-roll, disagreement, sympathy, sarcasm, or mockery",
        "- specific enough that the meme reply would feel justified",
        "- it should read like a normal tweet even if the meme reply never appears",
        "",
        "What the output should NOT be like:",
        "- not a caption for the meme",
        "- not a description of the image",
        "- not written from the perspective of someone posting the meme",
        "- not a punchline that already sounds like the meme itself",
        "- not overly internet-meme-styled unless it would genuinely fit a normal tweet",
        "- not stuffed with hashtags, emojis, or meme slang",
        "- do not mention the meme, image, reply, or reaction image",
        "- do not write like a setup line from a meme template",
        "",
        "Prefer tweets that sound like one of these:",
        "- a complaint",
        "- a bad take",
        "- an overconfident opinion",
        "- an awkward confession",
        "- a relatable frustration",
        "- a small brag",
        "- an excuse",
        "- a misunderstanding",
        "- a naive claim",
        "",
        "Return only the tweet text.",
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
        f"Now generate candidate {candidate_num}."
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


def call_openrouter(prompt: str, image_url: Optional[str], temperature: float) -> str:
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
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)

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
    return data["choices"][0]["message"]["content"].strip()


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
    print(f"Writing to {OUTPUT_JSONL_PATH_1}")
    if GENERATE_TWO_CANDIDATES:
        print(f"Writing to {OUTPUT_JSONL_PATH_2}")

    out1 = open(OUTPUT_JSONL_PATH_1, "w", encoding="utf-8")
    out2 = open(OUTPUT_JSONL_PATH_2, "w", encoding="utf-8") if GENERATE_TWO_CANDIDATES else None

    try:
        for idx in range(START_INDEX, end):
            example = rows[idx]
            image_url = safe_str(example.get("url"))
            image_url = image_url if looks_like_url(image_url) else None

            # Candidate 1
            try:
                prompt1 = build_prompt(example, candidate_num=1)
                tweet1 = call_openrouter(
                    prompt=prompt1,
                    image_url=image_url,
                    temperature=TEMPERATURE_CANDIDATE_1,
                )
                tweet1 = clean_output(tweet1)

                record1 = {
                    "index": idx,
                    "post_id": safe_str(example.get("post_id")),
                    "title": safe_str(example.get("title")),
                    "image_url": image_url,
                    "tweet_that_meme_replies_to": tweet1,
                    "candidate_id": 1,
                    "source_example": example,
                }
                print(f"[OK-1] {idx}: {tweet1}")
            except Exception as e:
                record1 = {
                    "index": idx,
                    "post_id": safe_str(example.get("post_id")),
                    "title": safe_str(example.get("title")),
                    "image_url": image_url,
                    "candidate_id": 1,
                    "error": str(e),
                    "source_example": example,
                }
                print(f"[ERR-1] {idx}: {e}")

            out1.write(json.dumps(record1, ensure_ascii=False) + "\n")

            # Candidate 2
            if GENERATE_TWO_CANDIDATES:
                try:
                    prompt2 = build_prompt(example, candidate_num=2)
                    tweet2 = call_openrouter(
                        prompt=prompt2,
                        image_url=image_url,
                        temperature=TEMPERATURE_CANDIDATE_2,
                    )
                    tweet2 = clean_output(tweet2)

                    record2 = {
                        "index": idx,
                        "post_id": safe_str(example.get("post_id")),
                        "title": safe_str(example.get("title")),
                        "image_url": image_url,
                        "tweet_that_meme_replies_to": tweet2,
                        "candidate_id": 2,
                        "source_example": example,
                    }
                    print(f"[OK-2] {idx}: {tweet2}")
                except Exception as e:
                    record2 = {
                        "index": idx,
                        "post_id": safe_str(example.get("post_id")),
                        "title": safe_str(example.get("title")),
                        "image_url": image_url,
                        "candidate_id": 2,
                        "error": str(e),
                        "source_example": example,
                    }
                    print(f"[ERR-2] {idx}: {e}")

                out2.write(json.dumps(record2, ensure_ascii=False) + "\n")

            time.sleep(SLEEP_SECONDS)

    finally:
        out1.close()
        if out2:
            out2.close()

    print("Done.")



if __name__ == "__main__":
    main()
