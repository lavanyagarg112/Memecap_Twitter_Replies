# Meme Reply Ranking Training

## Overview

This training code learns to rerank a fixed set of meme candidates for each tweet.

Each training task is:

- one `tweet_text`
- a fixed set of candidate memes from the CSV
- a per-candidate `rank` label where lower is better

The retrieval step is assumed to have already happened offline when the dataset was built. The code in this folder does not retrieve from a full meme bank at runtime. It only reranks the candidate set already present in the CSV.

## Quick Start

Run these from the repository root:

1. Install dependencies:

```bash
pip install -r training/requirements.txt
```

2. Download images once:

```bash
python training/download_images.py
```

3. Run a quick smoke test for each pipeline:

```bash
python training/train.py --pipeline text --encoder_type hf --device cuda --batch_size 16 --num_epochs 1 --save_dir training/checkpoints/text_hf_smoke
```

```bash
python training/train.py --pipeline image --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --num_epochs 1 --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/image_qwen_smoke
```

```bash
python training/train.py --pipeline multimodal --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --num_epochs 1 --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/multimodal_qwen_smoke
```

4. Launch the full runs after the smoke tests pass:

```bash
python training/train.py --pipeline text --encoder_type hf --device cuda --batch_size 16 --save_dir training/checkpoints/text_hf_clean
```

```bash
python training/train.py --pipeline image --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/image_qwen_clean
```

```bash
python training/train.py --pipeline multimodal --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/multimodal_qwen_clean
```

5. Evaluate the best checkpoint:

```bash
python training/eval.py --checkpoint training/checkpoints/text_hf_clean/best.pt --split test --device cuda
```

## Supported Pipelines

The current codebase is configured for these three experiment settings:

1. `text`
   tweet text + candidate text
   supported encoders: `hf`, `bow_mean`, `gru`

2. `image`
   tweet text + candidate image
   supported encoder: `qwen_vl`

3. `multimodal`
   tweet text + candidate image + candidate text
   supported encoder: `qwen_vl`

For the Qwen pipelines, the code uses `Qwen/Qwen2.5-VL-3B-Instruct` as a cross-encoder and produces one score per candidate meme.

## Default Dataset

By default, training uses the clean non-annotation split:

- `training/data/non-annotation-dataset/clean/train_clean.csv`
- `training/data/non-annotation-dataset/clean/val_clean.csv`
- `training/data/non-annotation-dataset/clean/test_clean.csv`

The expected row-level columns include:

- `task_id`
- `tweet_text`
- `meme_post_id`
- `image_url`
- `meme_title`
- `img_captions`
- `meme_captions`
- `metaphors`
- `rank`

Candidate text is built by joining:

- `meme_title`
- `img_captions`
- `meme_captions`
- `metaphors`

The full curated candidate set in the CSV is used as-is. Runtime candidate truncation is disabled.

## Image Download

For the image-only and multimodal pipelines, download the meme images once before training:

```bash
python training/download_images.py
```

By default, the downloader:

- reads the clean non-annotation train, val, and test CSVs
- downloads images into `training/data/non-annotation-dataset/images`
- logs failures to `training/data/non-annotation-dataset/images/download_failures.csv`

If you hit rate limits, use a slower run:

```bash
python training/download_images.py \
  --max_retries 8 \
  --retry_sleep 3 \
  --throttle 0.5 \
  --progress_every 50
```

Missing images are not fatal. If an image file cannot be found or loaded, the dataset loader falls back to a black placeholder image.

## Installation

Install the Python dependencies from the repo root:

```bash
pip install -r training/requirements.txt
```

## Training Commands

Run these from the repository root.

### Pipeline 1: Text

Hugging Face text encoder:

```bash
python training/train.py \
  --pipeline text \
  --encoder_type hf \
  --device cuda \
  --batch_size 16 \
  --save_dir training/checkpoints/text_hf_clean
```

Lightweight baselines:

```bash
python training/train.py \
  --pipeline text \
  --encoder_type bow_mean \
  --device cuda \
  --batch_size 16 \
  --save_dir training/checkpoints/text_bow_clean
```

```bash
python training/train.py \
  --pipeline text \
  --encoder_type gru \
  --device cuda \
  --batch_size 16 \
  --save_dir training/checkpoints/text_gru_clean
```

### Pipeline 2: Image

Recommended starting point:

```bash
python training/train.py \
  --pipeline image \
  --encoder_type qwen_vl \
  --device cuda \
  --batch_size 1 \
  --freeze_encoder \
  --image_dir training/data/non-annotation-dataset/images \
  --save_dir training/checkpoints/image_qwen_clean
```

### Pipeline 3: Multimodal

Recommended starting point:

```bash
python training/train.py \
  --pipeline multimodal \
  --encoder_type qwen_vl \
  --device cuda \
  --batch_size 1 \
  --freeze_encoder \
  --image_dir training/data/non-annotation-dataset/images \
  --save_dir training/checkpoints/multimodal_qwen_clean
```

### Smoke Tests

For a first sanity run, add:

- `--num_epochs 1`

For the Qwen pipelines, start with:

- `--batch_size 1`
- `--freeze_encoder`

Then increase only if your GPU memory allows it.

## Evaluation

Evaluate a saved checkpoint on validation or test:

```bash
python training/eval.py \
  --checkpoint training/checkpoints/text_hf_clean/best.pt \
  --split test \
  --device cuda
```

```bash
python training/eval.py \
  --checkpoint training/checkpoints/image_qwen_clean/best.pt \
  --split test \
  --device cuda
```

```bash
python training/eval.py \
  --checkpoint training/checkpoints/multimodal_qwen_clean/best.pt \
  --split test \
  --device cuda
```

Use `--split val` if you want validation metrics instead.

## Loss and Metrics

Default training loss:

- `bpr`

Optional training loss:

- `hinge`

Current evaluation metrics:

- `recall_at_1`
- `mrr`
- `ndcg_10`
- `score_at_1`

Best-checkpoint selection is based on validation `recall_at_1`.

## Checkpoints

Each run saves:

- `latest.pt`
- `best.pt`

Checkpoints include:

- model weights
- optimizer state
- scheduler state
- serialized config
- serialized vocab

## Relevant Files

- `training/config.py`
  CLI and dataclass configuration

- `training/dataset.py`
  CSV loading, grouped task construction, image loading, collate logic

- `training/model.py`
  Text encoders and Qwen cross-encoder ranking model

- `training/losses.py`
  Pairwise BPR and hinge ranking losses

- `training/metrics.py`
  Ranking metrics used during validation and test evaluation

- `training/train.py`
  Main training entry point

- `training/eval.py`
  Checkpoint evaluation entry point

- `training/download_images.py`
  One-time image downloader for the clean non-annotation split

## Notes

- `inference.py` is not part of the documented training workflow here.
- If you train on CPU, expect the Qwen pipelines to be very slow.
- If training fails at startup, first verify your Python environment, CUDA setup, and image directory path.
