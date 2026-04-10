"""
Each dataset item = one task = one tweet + K candidate memes.

Image handling priority (for image / multimodal pipelines):
  1. Local file at  <image_dir>/<meme_post_id>.{jpg,jpeg,png,webp}
  2. Download from  image_url  at runtime  (requires requests; slow)
  3. Black 224x224 placeholder            (silent fallback)
"""

from __future__ import annotations

import csv
import io
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


@dataclass
class _Candidate:
    meme_post_id:   str
    image_url:      str
    candidate_text: str
    rank:           int
    image:          Optional[Image.Image] = None


@dataclass
class _Task:
    task_id:    str
    tweet_text: str
    candidates: List[_Candidate]


@dataclass
class Batch:
    """
    All tensors that a model forward pass may need.

    context_input_ids      : [B, Lc]         tweet token ids
    context_attention_mask : [B, Lc]
    candidate_input_ids    : [B, K, Lm]      candidate text token ids   (None = image-only)
    candidate_attention_mask: [B, K, Lm]                                (None = image-only)
    pixel_values           : [B, K, C, H, W] candidate images for non-Qwen image paths
                           : [sum(image_patches), patch_dim] for Qwen-VL
    candidate_mask         : [B, K]          1 = real candidate, 0 = pad slot
    ranks                  : [B, K]          int  1 = best, higher = worse
    task_ids               : List[str]       length B
    """
    context_input_ids:       Optional[torch.Tensor]
    context_attention_mask:  Optional[torch.Tensor]
    candidate_input_ids:     Optional[torch.Tensor]
    candidate_attention_mask: Optional[torch.Tensor]
    pixel_values:            Optional[torch.Tensor]
    image_grid_thw:          Optional[torch.Tensor]
    candidate_mask:          torch.Tensor    # [B, K]  real=1, pad=0
    ranks:                   torch.Tensor    # [B, K]
    task_ids:                List[str]


def _build_qwen_pair_prompt(
    processor,
    tweet_text: str,
    candidate_text: str = "",
) -> str:
    prompt = [f"Tweet:\n{tweet_text}"]
    if candidate_text.strip():
        prompt.append(f"Candidate meme text:\n{candidate_text}")
        prompt.append(
            "How suitable is this meme image as a reply to the tweet?"
        )
    else:
        prompt.append("How suitable is this meme image as a reply to the tweet?")
    prompt_text = "\n\n".join(prompt)
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return prompt_text

def _find_local_image(image_dir: str, meme_post_id: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join(image_dir, meme_post_id + ext)
        if os.path.exists(p):
            return p
    return None


def _download_image(url: str, retries: int = 2) -> Optional[Image.Image]:
    if not (_HAS_REQUESTS and url):
        return None
    for attempt in range(retries):
        try:
            r = _requests.get(url, timeout=10)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def _load_image(meme_post_id: str, image_url: str, image_dir: str) -> Image.Image:
    """
    Try local --> download --> black placeholder.
    """
    if image_dir:
        local = _find_local_image(image_dir, meme_post_id)
        if local:
            try:
                return Image.open(local).convert("RGB")
            except Exception:
                pass

    img = _download_image(image_url)
    if img is not None:
        return img

    return Image.new("RGB", (224, 224), color=(0, 0, 0))


def _load_tasks(csv_path: str, candidate_text_fields: List[str]) -> List[_Task]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    grouped: Dict[str, _Task] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["task_id"]
            if tid not in grouped:
                grouped[tid] = _Task(
                    task_id    = tid,
                    tweet_text = row["tweet_text"],
                    candidates = [],
                )
            parts = [row.get(fld, "").strip() for fld in candidate_text_fields]
            cand_text = " | ".join(p for p in parts if p)
            grouped[tid].candidates.append(
                _Candidate(
                    meme_post_id   = row["meme_post_id"],
                    image_url      = row.get("image_url", ""),
                    candidate_text = cand_text,
                    rank           = int(row["rank"]),
                )
            )

    for task in grouped.values():
        task.candidates.sort(key=lambda c: c.rank)
    return list(grouped.values())


class MemeDataset(Dataset):
    def __init__(
        self,
        tasks:          List[_Task],
        pipeline:       str,
        image_dir:      str  = "",
        min_candidates: int  = 2,
    ):
        self.pipeline  = pipeline
        self.image_dir = image_dir
        need_images    = pipeline in ("image", "multimodal")
        filtered       = []

        for task in tasks:
            cands = task.candidates
            if len(cands) < min_candidates:
                continue
            if need_images:
                for c in cands:
                    c.image = _load_image(c.meme_post_id, c.image_url, image_dir)
            filtered.append(_Task(task.task_id, task.tweet_text, cands))

        self.tasks = filtered
        print(f"[Dataset] {len(self.tasks)} tasks  pipeline={pipeline}")

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> _Task:
        return self.tasks[idx]


def make_collate_fn(
    pipeline:        str,
    encoder_type:    Optional[str] = None,
    tokenizer=None,         # HF AutoTokenizer or CLIPProcessor
    image_processor=None,   # HF CLIPImageProcessor (when not using full CLIPProcessor)
    vocab=None,             # Vocab instance (BOW / GRU path)
    text_cfg=None,          # TextConfig
):
    is_qwen_vl = encoder_type == "qwen_vl" and pipeline in ("image", "multimodal")
    need_text  = pipeline in ("text", "multimodal")
    need_image = pipeline in ("image", "multimodal")
    max_ctx    = text_cfg.max_context_len if text_cfg else 128
    max_cand   = text_cfg.max_cand_len    if text_cfg else 128

    def collate_fn(tasks: List[_Task]) -> Batch:
        B = len(tasks)
        K = max(len(t.candidates) for t in tasks)

        ctx_texts    = [t.tweet_text for t in tasks]
        cand_texts   = []   # length B*K (padded slots get "")
        pair_texts   = []   # length B*K for Qwen image mode
        images       = []   # length B*K
        ranks        = torch.full((B, K), fill_value=K + 1, dtype=torch.long)
        cand_mask    = torch.zeros(B, K, dtype=torch.long)

        for b, task in enumerate(tasks):
            for k, cand in enumerate(task.candidates):
                cand_texts.append(cand.candidate_text)
                if is_qwen_vl:
                    pair_texts.append(
                        _build_qwen_pair_prompt(
                            tokenizer,
                            task.tweet_text,
                            cand.candidate_text if pipeline == "multimodal" else "",
                        )
                    )
                if need_image:
                    images.append(
                        cand.image if cand.image is not None
                        else Image.new("RGB", (224, 224))
                    )
                ranks[b, k]     = cand.rank
                cand_mask[b, k] = 1
            for k in range(len(task.candidates), K):
                cand_texts.append("")
                if is_qwen_vl:
                    pair_texts.append(
                        _build_qwen_pair_prompt(
                            tokenizer,
                            task.tweet_text,
                            "",
                        )
                    )
                if need_image:
                    images.append(Image.new("RGB", (224, 224)))

        ctx_ids = ctx_mask = cand_ids = cand_attn = None
        pixel_values = image_grid_thw = None

        if is_qwen_vl:
            out = tokenizer(
                text=pair_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            cand_ids  = out["input_ids"].view(B, K, -1)
            cand_attn = out["attention_mask"].view(B, K, -1)
            pixel_values = out["pixel_values"]
            if "image_grid_thw" in out:
                image_grid_thw = out["image_grid_thw"]

        elif need_text and tokenizer is not None:
            # HuggingFace tokenizer (AutoTokenizer or CLIPProcessor)
            ctx_enc = tokenizer(
                ctx_texts,
                padding=True, truncation=True, max_length=max_ctx,
                return_tensors="pt",
            )
            cand_enc = tokenizer(
                cand_texts,
                padding=True, truncation=True, max_length=max_cand,
                return_tensors="pt",
            )
            ctx_ids  = ctx_enc["input_ids"]                         # [B, Lc]
            ctx_mask = ctx_enc["attention_mask"]
            cand_ids  = cand_enc["input_ids"].view(B, K, -1)        # [B, K, Lm]
            cand_attn = cand_enc["attention_mask"].view(B, K, -1)

        elif need_text and vocab is not None:
            # BOW / GRU path
            ctx_ids,  ctx_mask  = vocab.encode_batch(ctx_texts,  max_ctx)
            flat_ids, flat_mask = vocab.encode_batch(cand_texts, max_cand)
            cand_ids  = flat_ids.view(B, K, -1)
            cand_attn = flat_mask.view(B, K, -1)

        if need_image and images and not is_qwen_vl:
            proc = image_processor if image_processor is not None else tokenizer
            if proc is not None:
                out = proc(images=images, return_tensors="pt")
                pv  = out["pixel_values"]                   # [B*K, C, H, W]
                pixel_values = pv.view(B, K, *pv.shape[1:])

        return Batch(
            context_input_ids        = ctx_ids,
            context_attention_mask   = ctx_mask,
            candidate_input_ids      = cand_ids,
            candidate_attention_mask = cand_attn,
            pixel_values             = pixel_values,
            image_grid_thw           = image_grid_thw,
            candidate_mask           = cand_mask,
            ranks                    = ranks,
            task_ids                 = [t.task_id for t in tasks],
        )

    return collate_fn

collate_fn = None


def load_datasets(config):
    from text_utils import Vocab, build_vocab

    cfg      = config
    pipeline = cfg.model.pipeline
    enc_type = cfg.model.encoder_type

    def _tasks(path):
        return _load_tasks(path, cfg.data.candidate_text_fields)

    train_tasks = _tasks(cfg.data.train_csv)
    val_tasks   = _tasks(cfg.data.val_csv)
    test_tasks  = _tasks(cfg.data.test_csv)

    if enc_type in ("bow_mean", "gru"):
        all_texts = []
        for task in train_tasks:
            all_texts.append(task.tweet_text)
            for c in task.candidates:
                all_texts.append(c.candidate_text)
        vocab = build_vocab(all_texts, max_size=cfg.model.vocab_size)
    else:
        vocab = Vocab(max_size=2)

    tokenizer, image_processor = _build_processors(cfg)

    global collate_fn
    collate_fn = make_collate_fn(
        pipeline        = pipeline,
        encoder_type    = enc_type,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        vocab           = vocab,
        text_cfg        = cfg.text,
    )

    def _ds(tasks):
        return MemeDataset(
            tasks          = tasks,
            pipeline       = pipeline,
            image_dir      = cfg.data.image_dir,
            min_candidates = cfg.data.min_candidates_per_task,
        )

    return _ds(train_tasks), _ds(val_tasks), _ds(test_tasks), vocab


def _build_processors(cfg):
    """
    Returns (tokenizer, image_processor).

    encoder_type | pipeline  | tokenizer               | image_processor
    ─────────────────────────────────────────────────────────────────────
    hf           | text       | AutoTokenizer           | None
    hf           | multimodal | AutoTokenizer           | CLIPImageProcessor
    clip         | *          | CLIPProcessor           | CLIPProcessor
    qwen_vl      | image/multimodal | AutoProcessor      | AutoProcessor
    bow_mean/gru | *          | None (use Vocab)        | CLIPImageProcessor?
    llava        | multimodal | LlavaProcessor          | LlavaProcessor
    """
    enc  = cfg.model.encoder_type
    pipe = cfg.model.pipeline

    tokenizer       = None
    image_processor = None

    if enc == "hf":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_model_name)
        if pipe in ("image", "multimodal"):
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(cfg.model.clip_model_name)

    elif enc == "clip":
        from transformers import CLIPProcessor
        proc = CLIPProcessor.from_pretrained(cfg.model.clip_model_name)
        tokenizer       = proc
        image_processor = proc

    elif enc == "qwen_vl":
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(cfg.model.qwen_vl_model_name)
        tokenizer       = proc
        image_processor = proc

    elif enc == "llava":
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(cfg.model.llava_model_name)
        tokenizer       = proc
        image_processor = proc

    elif enc in ("bow_mean", "gru"):
        # vocab handles text; still need image processor for image pipelines
        if pipe in ("image", "multimodal"):
            from transformers import CLIPImageProcessor
            image_processor = CLIPImageProcessor.from_pretrained(cfg.model.clip_model_name)

    return tokenizer, image_processor
