# Pre-training: Data Collection & Annotation Pipeline

## Non-Annotation Pipeline (NEW)

Alternative pipeline using real tweets from [HSDSLab/TwitterMemes](https://huggingface.co/datasets/HSDSLab/TwitterMemes) instead of synthetic ones. Uses a VLM to describe each tweet+meme, then ranks the top 10 most semantically similar MemeCap memes via embedding similarity. No human annotation needed.

```bash
cd non-annotation
pip install datasets Pillow
python rank_similar_memes.py --limit 500 --dry-run   # check cost
python rank_similar_memes.py --limit 500              # run (~$0.10)
python rank_similar_memes.py --rerank                 # optional VLM re-ranking
python view_rankings.py                               # visualise results (localhost:5002)
```

Outputs `train.csv`, `val.csv`, `test.csv` in the same format as the annotation pipeline. Flagged memes from `flagged_memes.json` are automatically excluded.

---

## Setup

```bash
pip install requests numpy sentence-transformers flask python-dotenv
```

Create `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

## Pipeline Steps

### 1. Generate Synthetic Tweets

**Direct style** (tweet closely matches meme context):
```bash
python create_train_context_3.py
```

**Indirect style** (tweet shares emotional energy, different topic):
```bash
python create_train_context_4.py
```

Both generate 1 tweet per meme from MemeCap trainval split (1500 memes).

### 2. Clean Tweet Files

Removes errors, keeps both files balanced:
```bash
python clean_tweets.py
```

### 3. Flag Offensive Memes

Interactive review — scans for offensive words, you decide which to exclude:
```bash
python flag_memes.py
```
Progress saved to `flagged_memes.json`. Quit with `q` and resume later.

### 4. Select Candidate Memes

For each tweet, selects 9 distractors (4 semantic + 5 random) plus the original:
```bash
python select_candidates_v2.py
```

### 5. Human Annotation

```bash
cd annotation_app
python app.py
```
Open `http://localhost:5000`. Annotators judge "does this meme work as a reply?" (Yes/No). Admin dashboard at `/admin`.

### 6. Model Annotation

Augments human annotations with 3 VLM models (seed-1.6-flash, gemini-lite, gemini-3-flash). Each item gets 3 total annotations (human + model combined). Broken images skipped.

```bash
python annotate_parallel.py --dry-run   # preview cost & count
python annotate_parallel.py             # run (~1 hour, 10 workers)
python annotate_parallel.py --budget 5  # custom budget in USD
```

Outputs: `annotations_augmented.csv`, `meme_rankings.csv`

### 7. Generate Training Data

```bash
python create_train_data.py
```

Outputs train/val/test CSVs (80/10/10 split by task) in both full and clean (flagged memes removed) versions.

## Key Files

| File | Description |
|------|-------------|
| `annotations.csv` | Human annotations export |
| `annotations_augmented.csv` | Human + model annotations |
| `meme_rankings.csv` | Candidates ranked by avg score per tweet |
| `annotation_tasks_clean.json` | All 2,718 tasks with 10 candidates each |
| `flagged_memes.json` | Offensive meme review results |
| `validation_results.csv` | Model validation against human labels |
