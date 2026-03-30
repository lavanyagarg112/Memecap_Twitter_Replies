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
├── training/                   # Model training
│   ├── data/
│   │   ├── train.csv, val.csv, test.csv
│   │   └── clean/
│   │       └── train_clean.csv, val_clean.csv, test_clean.csv
│   └── init.py
└── README.md
```

## Pipelines

### Pipeline A: Non-Annotation (primary)

Uses real tweets from [HSDSLab/TwitterMemes](https://huggingface.co/datasets/HSDSLab/TwitterMemes). A VLM describes each tweet+meme, then the top 10 most semantically similar MemeCap memes are ranked via embedding similarity. No human annotation required.

```bash
cd pre-training/non-annotation
python rank_similar_memes.py --limit 500    # ~$0.10, ~15 min
python view_rankings.py                     # visualise at localhost:5002
```

See [`pre-training/README.md`](pre-training/README.md) for full details.

### Pipeline B: Annotation-based

Synthetic tweets + human/VLM annotation. Scripts in `pre-training/`. See [`pre-training/README.md`](pre-training/README.md).

```
Generate tweets → Clean → Flag memes → Select candidates → Human annotation → Model annotation → Rankings → Train data
```

### Training Data

Both pipelines output pointwise format — one row per tweet-meme pair. Shared columns:

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
