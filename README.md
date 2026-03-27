# Meme Reply Selection Pipeline

Trains a model to rank memes as replies to tweets. Given a tweet, the model learns which memes work as good replies based on human + model annotations.

## Project Structure

```
project/
├── pre-training/       # Data collection & annotation pipeline
│   ├── annotation_app/ # Flask app for human annotations
│   ├── annotate_parallel.py   # Multi-model VLM annotation (parallel)
│   ├── annotate_with_models.py # Multi-model VLM annotation (sequential)
│   ├── create_train_data.py   # Generate train/val/test splits
│   ├── annotations.csv        # Human annotations
│   ├── annotations_augmented.csv # Human + model annotations
│   └── meme_rankings.csv      # Ranked meme candidates per tweet
├── training/           # Model training
│   ├── data/
│   │   ├── train.csv, val.csv, test.csv          # Full dataset
│   │   └── clean/
│   │       └── train_clean.csv, val_clean.csv, test_clean.csv  # Flagged memes removed
│   └── init.py
└── README.md
```

## Pipeline

### 1. Pre-training (data collection)

All scripts in `pre-training/`. See [`pre-training/README.md`](pre-training/README.md) for detailed steps.

```
Generate tweets → Clean → Flag memes → Select candidates → Human annotation → Model annotation → Rankings → Train data
```

### 2. Annotation

Human annotators judge meme-tweet pairs ("does this meme work as a reply?"). To augment limited human annotations, 3 VLM models (seed-1.6-flash, gemini-lite, gemini-3-flash) simulate additional annotators.

Each item gets 3 total annotations (human + model combined). Broken images are skipped.

### 3. Training Data

Pointwise format — one row per tweet-meme pair:

| Column | Description |
|--------|-------------|
| `tweet_text` | The tweet |
| `meme_title` | Meme title |
| `img_captions` | MemeCap image descriptions |
| `meme_captions` | MemeCap meme meaning |
| `metaphors` | MemeCap visual metaphors |
| `selection_method` | original / semantic / random |
| `avg_score` | Mean annotation score (0-1) |
| `rank` | Rank within the tweet's candidates (1 = best) |

Two versions: full (all memes) and clean (user-flagged inappropriate memes removed).

Split: 80/10/10 by task (all candidates for a tweet stay in the same split).

## Setup

```bash
pip install requests python-dotenv
export OPENROUTER_API_KEY=your_key
```

## Quick Run

```bash
# Annotate (from pre-training/)
cd pre-training
python annotate_parallel.py --dry-run   # preview
python annotate_parallel.py             # run (~1 hour)

# Generate training data
python create_train_data.py
```
