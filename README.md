# Meme Reply Selection Pipeline

Trains a model to rank memes as replies to tweets. Given a tweet, the model learns which memes work as good replies based on human + model annotations.

## Project Structure

```
project/
├── pre-training/               # Data collection pipelines
│   ├── annotation_app/         # Flask app for human annotations
│   ├── non-annotation/         # VLM similarity pipeline (no annotation needed)
│   │   ├── rank_similar_memes.py   # Main pipeline
│   │   └── view_rankings.py        # Visualise results
│   ├── annotate_parallel.py    # Multi-model VLM annotation (parallel)
│   ├── create_train_data.py    # Generate train/val/test splits
│   └── meme_rankings.csv       # Ranked meme candidates per tweet
├── training/                   # Reranker training
│   ├── data/non-annotation-dataset/clean/
│   │   ├── train_clean.csv
│   │   ├── val_clean.csv
│   │   └── test_clean.csv
│   ├── download_images.py      # One-time image downloader
│   ├── train.py                # Training entry point
│   ├── eval.py                 # Checkpoint evaluation
│   └── README.md               # Full training instructions
└── README.md
```

## Pipelines

### Pipeline A: Non-Annotation (primary)

Uses real tweets from [HSDSLab/TwitterMemes](https://huggingface.co/datasets/HSDSLab/TwitterMemes). A VLM (`seed-1.6-flash`) describes each tweet+meme, then `all-MiniLM-L6-v2` builds a candidate pool via embedding similarity. The same VLM selects the top 10 from the pool. An optional final pass (`--rerank`) uses `gemini-3-flash` to rank those 10. No human annotation required.

```bash
cd pre-training/non-annotation
python rank_similar_memes.py --limit 500    # ~$0.10, ~15 min
python rank_similar_memes.py --rerank       # with Gemini ranking pass
python view_rankings.py                     # visualise at localhost:5002
```

See [`pre-training/README.md`](pre-training/README.md) for full details.

### Pipeline B: Annotation-based

Synthetic tweets + human/VLM annotation. Scripts in `pre-training/`. See [`pre-training/README.md`](pre-training/README.md).

```
Generate tweets → Clean → Flag memes → Select candidates → Human annotation → Model annotation → Rankings → Train data
```

### Training Data

Both pipelines output one row per tweet-meme candidate pair. Training groups rows by `task_id` and reranks the fixed candidate set already stored in the CSV.

Shared columns:

| Column | Description |
|--------|-------------|
| `tweet_text` | The tweet |
| `meme_title` | Meme title |
| `img_captions` | MemeCap image descriptions |
| `meme_captions` | MemeCap meme meaning |
| `metaphors` | MemeCap visual metaphors |
| `rank` | Rank within the tweet's candidates (1 = best) |

Pipeline-specific columns:

| Column | Pipeline A (non-annotation) | Pipeline B (annotation) |
|--------|---------------------------|------------------------|
| `selection_method` | `vlm_similarity` / `vlm_reranked` | `original` / `semantic` / `random` |
| scoring | `similarity_score` (cosine, 0-1) | `avg_score` (mean of yes/no votes, 0-1) |
| vote counts | — | `num_votes`, `num_yes`, `num_no` |

Split: 80/10/10 by task. Flagged inappropriate memes automatically excluded.

## Training

The training code currently assumes retrieval is already done offline. It uses the fixed candidate sets in:

- `training/data/non-annotation-dataset/clean/train_clean.csv`
- `training/data/non-annotation-dataset/clean/val_clean.csv`
- `training/data/non-annotation-dataset/clean/test_clean.csv`

The three intended training runs are:

1. Text pipeline
   tweet + candidate text

2. Image pipeline
   tweet + candidate image

3. Multimodal pipeline
   tweet + candidate image + candidate text

Image and multimodal training use `Qwen/Qwen2.5-VL-3B-Instruct` as the VLM cross-encoder.

### Training Quick Start

From the repo root:

```bash
pip install -r training/requirements.txt
python training/download_images.py
```

Smoke tests:

```bash
python training/train.py --pipeline text --encoder_type hf --device cuda --batch_size 16 --num_epochs 1 --save_dir training/checkpoints/text_hf_smoke
```

```bash
python training/train.py --pipeline image --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --num_epochs 1 --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/image_qwen_smoke
```

```bash
python training/train.py --pipeline multimodal --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --num_epochs 1 --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/multimodal_qwen_smoke
```

Full runs:

```bash
python training/train.py --pipeline text --encoder_type hf --device cuda --batch_size 16 --save_dir training/checkpoints/text_hf_clean
```

```bash
python training/train.py --pipeline image --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/image_qwen_clean
```

```bash
python training/train.py --pipeline multimodal --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/multimodal_qwen_clean
```

Evaluation:

```bash
python training/eval.py --checkpoint training/checkpoints/text_hf_clean/best.pt --split test --device cuda
```

See [`training/README.md`](training/README.md) for the full training workflow.

## Setup

```bash
pip install requests python-dotenv
export OPENROUTER_API_KEY=your_key
```

## Quick Run

```bash
# Non-annotation pipeline (recommended)
cd pre-training/non-annotation
python rank_similar_memes.py --limit 500 --dry-run
python rank_similar_memes.py --limit 500

# Or annotation pipeline
cd pre-training
python annotate_parallel.py --dry-run
python annotate_parallel.py
python create_train_data.py
```
