# Meme Reply Selection with PyTorch

## Overview

This project trains and evaluates lightweight PyTorch models for **meme reply selection**.

The task is a **grouped ranking problem**: given a tweet or post context and a set of candidate memes, the model assigns a scalar score to each candidate and selects the highest-scoring meme reply.

Formally, for a context `c` and candidate memes `M`, the model learns a scoring function:

`score(c, m)`

and predicts:

`argmax_m score(c, m)`

The implementation is inspired by the paper *Memes-as-Replies: Can Models Select Humorous Manga Panel Responses?* and follows its framing as closely as practical for this dataset and training setup. The codebase supports:

- **similarity-based ranking**
- **preference-based ranking**

The primary evaluation metric is **Score@1**, which measures the average human `avg_score` of the meme selected by the model.

## Repository Structure

```text
.
├── config.py          # Dataclass-based configuration and CLI parsing
├── text_utils.py      # Text normalization, tokenization, vocabulary, encoding
├── dataset.py         # CSV loading, task grouping, dataset objects, collate_fn
├── model.py           # TextEncoder, SimilarityRanker, PreferenceRanker
├── losses.py          # Pairwise hinge loss, BPR loss, loss dispatcher
├── metrics.py         # Score@1, Recall@1, MRR, nDCG@k, consensus hit rate
├── utils.py           # Seeding, device transfer, checkpoint save/load, meters
├── train.py           # Training loop, validation, checkpointing, final test eval
├── eval.py            # Checkpoint evaluation on val or test split
├── inference.py       # Rank candidate memes for a single input context
└── data/
    ├── clean/
    │   ├── train_clean.csv
    │   ├── val_clean.csv
    │   └── test_clean.csv
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### File roles

- `config.py`: centralizes all runtime settings in dataclasses.
- `text_utils.py`: handles preprocessing with a simple whitespace tokenizer and train-split vocabulary.
- `dataset.py`: preserves grouped task structure by `task_id` and builds padded batch tensors.
- `model.py`: contains the ranking models and shared text encoder options.
- `losses.py`: converts task-local rank order into pairwise ranking supervision.
- `metrics.py`: implements paper-style evaluation metrics centered on Score@1.
- `train.py`: full training entry point with validation and checkpoint selection.
- `eval.py`: evaluates a saved checkpoint on `val` or `test`.
- `inference.py`: ranks candidate meme rows for a new context using the same preprocessing path as training.

## Dataset Format

The code expects clean CSV splits with one row per `(context, candidate meme)` pair.

### Default paths used by the code

The current implementation defaults to:

- `data/clean/train_clean.csv`
- `data/clean/val_clean.csv`
- `data/clean/test_clean.csv`

The dataset loader also supports the alternate layout:

- `data/train_clean.csv`
- `data/val_clean.csv`
- `data/test_clean.csv`

If the default path does not exist, the loader will automatically try the alternate location.

You can also override paths explicitly from the CLI:

```bash
python train.py \
  --train_csv data/train_clean.csv \
  --val_csv data/val_clean.csv \
  --test_csv data/test_clean.csv
```

### Important columns

The project uses columns such as:

- `task_id`
- `tweet_text`
- `meme_post_id`
- `meme_title`
- `img_captions`
- `meme_captions`
- `metaphors`
- `rank`
- `avg_score`
- `num_votes`
- `num_yes`
- `num_no`

### Label interpretation

- One `task_id` corresponds to **one grouped ranking task**.
- All rows that share the same `task_id` are candidate memes for the same context.
- `tweet_text` is the context text.
- Each row is a single candidate meme.
- Lower `rank` is better, with `rank = 1` representing the preferred candidate.
- `avg_score` is used as a graded human relevance / funniness signal for evaluation, including **Score@1**.

This project does **not** treat each row as an independent classification example.

## How the Pipeline Works

### 1. CSV loading
`dataset.py` reads the train, validation, and test CSV files using `csv.DictReader`.

### 2. Grouping by `task_id`
Rows are grouped into task-level examples so each training item contains:

- one context
- multiple candidate memes
- per-candidate labels and metadata

Tasks with fewer than `min_candidates_per_task` candidates are filtered out.

### 3. Candidate text construction
For each candidate meme, text is built by joining the configured fields:

- `meme_title`
- `img_captions`
- `meme_captions`
- `metaphors`

This produces a single candidate text string per row.

### 4. Tokenization and vocabulary
The project uses a simple whitespace tokenizer.

- Text is normalized with optional lowercasing.
- A vocabulary is fit on the **training split only**.
- Special tokens include PAD and UNK.
- Contexts and candidate texts are encoded separately.

### 5. Grouped batching
`collate_fn` pads each batch to:

- the maximum context length in the batch
- the maximum candidate text length in the batch
- the maximum number of candidates in the batch

The batch preserves task structure with tensors shaped like:

- context: `[B, Lc]`
- candidates: `[B, K, Lm]`

### 6. Candidate scoring
A model computes a score for every candidate in a task:

- `SimilarityRanker`: context-candidate embedding similarity
- `PreferenceRanker`: direct conditioned scoring with interaction features

### 7. Loss computation
Training uses pairwise ranking supervision derived from `rank`:

- candidate `i` is preferred over `j` if `rank_i < rank_j`

Supported losses:

- pairwise hinge loss
- BPR loss

### 8. Metric computation
Evaluation computes grouped retrieval/ranking metrics:

- Score@1
- Recall@1
- MRR
- nDCG@k
- consensus hit rate

### 9. Checkpointing
`train.py` saves:

- `latest.pt` after every epoch
- `best.pt` when validation **Score@1** improves

At the end of training, the best checkpoint is reloaded and evaluated on the test split.

## Model Variants

## Similarity-Based Ranker

The similarity model separately encodes the context and each candidate meme into embeddings, then scores them using either:

- cosine similarity
- dot product

This approximates the paper’s **similarity-based selection** setting.

Use it when you want a simple retrieval-style scoring function with fewer learned interaction parameters.

## Preference-Based Ranker

The preference model also encodes the context and candidate separately, but then builds richer interaction features:

- context embedding
- candidate embedding
- absolute difference
- elementwise product

These features are passed through an MLP to produce a scalar score for each candidate.

This approximates the paper’s **preference-based selection** setting and usually gives the model more flexibility than raw similarity alone.

## Training Objective

The training target is derived from **relative ranking within each task**, not from standalone row labels.

For any pair of candidates in the same task:

- candidate `i` is preferred over `j` if `rank_i < rank_j`

Supported objectives:

- **Hinge loss**: encourages preferred candidates to outscore lower-ranked candidates by a margin
- **BPR loss**: encourages the score difference between preferred and non-preferred candidates to be positive

This is a ranking problem because the model must choose the best meme **within a candidate set**, not assign an independent class label to each row.

## Evaluation Metrics

## Score@1
Primary metric.

For each task:

1. choose the candidate with the highest predicted score
2. look up that candidate’s human `avg_score`
3. average over tasks

This directly measures how good the model’s top choice is according to human judgments.

## Recall@1
Measures whether the top predicted candidate matches the target best item.

The target can be defined by:

- lowest `rank` (`rank1`)
- highest `avg_score` (`avg_score`)

## MRR
Mean Reciprocal Rank of the first correct target candidate in the model’s predicted ordering.

## nDCG@k
Uses `avg_score` as graded relevance and evaluates how well the predicted ranking matches human preference strength.

## Consensus Hit Rate
Measures how often the top predicted candidate has `avg_score >= threshold`, where the threshold is configurable.

## Installation

Create a virtual environment and install the minimal dependencies.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to Run

## Train a similarity-based model

```bash
python train.py --model_type similarity
```

## Train a preference-based model

```bash
python train.py --model_type preference
```

## Train using alternate CSV paths

If your clean splits are in `data/train_clean.csv`, `data/val_clean.csv`, and `data/test_clean.csv`:

```bash
python train.py \
  --train_csv data/train_clean.csv \
  --val_csv data/val_clean.csv \
  --test_csv data/test_clean.csv
```

## Evaluate a checkpoint

```bash
python eval.py --checkpoint checkpoints/best.pt --split test
```

Evaluate on the validation split instead:

```bash
python eval.py --checkpoint checkpoints/best.pt --split val
```

## Run inference

`inference.py` expects:

- a checkpoint
- a raw `tweet_text`
- a JSON file containing a list of candidate meme rows

Example:

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --tweet_text "My roommate asked me to help find his phone, but I think it's still in my pocket." \
  --candidates_json candidates.json
```

Optional top-k display:

```bash
python inference.py \
  --checkpoint checkpoints/best.pt \
  --tweet_text "Example context" \
  --candidates_json candidates.json \
  --top_k 3
```

### Example candidate JSON format

```json
[
  {
    "meme_post_id": "abc123",
    "meme_title": "Example meme title",
    "img_captions": "A character looks shocked.",
    "meme_captions": "Used when someone gets caught doing something awkward.",
    "metaphors": "character -> meme poster"
  },
  {
    "meme_post_id": "def456",
    "meme_title": "Another meme",
    "img_captions": "Two people staring at each other.",
    "meme_captions": "Used for silent tension.",
    "metaphors": "person -> poster"
  }
]
```

## Checkpoints and Outputs

By default, checkpoints are saved in:

- `checkpoints/latest.pt`
- `checkpoints/best.pt`

### Checkpoint behavior

- `latest.pt` is overwritten every epoch.
- `best.pt` is updated only when validation **Score@1** improves.

### What is stored

Each checkpoint includes:

- model weights
- optimizer state
- scheduler state
- epoch number
- best metric value
- serialized config
- serialized vocabulary

Model selection is based on **validation Score@1**.

## Configuration

Configuration is defined in `config.py` using dataclasses:

- `DataConfig`
- `TextConfig`
- `ModelConfig`
- `TrainConfig`
- `EvalConfig`
- `Config`

### Common settings to change

#### DataConfig
- CSV paths
- minimum candidates per task
- candidate text fields

#### TextConfig
- vocabulary size
- max context length
- max candidate length
- lowercasing behavior

#### ModelConfig
- `model_type`: `similarity` or `preference`
- `text_encoder_type`: `bow_mean` or `gru`
- embedding and hidden dimensions
- similarity function
- shared vs separate encoders

#### TrainConfig
- batch size
- number of epochs
- learning rate
- weight decay
- device
- loss type
- save directory

#### EvalConfig
- `ndcg_k`
- consensus threshold
- Recall@1 target type

Most users will start by changing:

- data paths
- `model_type`
- `text_encoder_type`
- batch size
- number of epochs
- learning rate
- device

## Extending the Project

### Change candidate text fields
The candidate representation is controlled by `candidate_text_fields` in `DataConfig`.

For example, you can change which metadata fields are concatenated into candidate text without changing the rest of the pipeline.

### Change the encoder
The text encoder currently supports:

- `bow_mean`
- `gru`

You can add a new encoder inside `TextEncoder` while preserving the same `[N, D]` output interface.

### Add image features later
The current implementation is text-only by default, which matches the lightweight setup and the paper’s observation that visual information does not always help consistently.

If you want to add image features later, the cleanest extension points are:

- dataset loading for image paths/features
- `Batch` structure
- model fusion layer

### Add listwise or score-aware training
The current training uses pairwise supervision derived from `rank`.

Possible future extensions:

- listwise ranking losses
- direct regression or calibration against `avg_score`
- uncertainty-aware weighting using `num_votes`
- weighted pairwise losses using agreement statistics

### Add retrieve-and-rerank
The code currently performs direct ranking over the provided candidate set.

A retrieve-and-rerank pipeline could be added by:

1. retrieving candidate memes from a large pool
2. passing top retrieved items into the existing ranker

That would fit naturally on top of the current grouped scoring interface.

## Troubleshooting / Notes

### Missing data files
If training fails at startup, verify that your CSV paths match the paths passed to `train.py`.

The default code paths are `data/clean/train_clean.csv`, `data/clean/val_clean.csv`, and `data/clean/test_clean.csv`.

### Empty tasks after filtering
If a split becomes empty, check:

- `task_id` values exist
- each task has at least `min_candidates_per_task`
- the CSV was parsed correctly

### CPU vs GPU
The default config uses `device="cuda"`, but the training code falls back to CPU if CUDA is unavailable.

You can also set:

```bash
python train.py --device cpu
```

### Checkpoint path problems
For evaluation and inference, make sure `--checkpoint` points to an existing `.pt` file such as `checkpoints/best.pt`.

The code was validated against Torch 2.6-style checkpoint loading, and checkpoints now load explicitly with full state restoration.

### Inference JSON shape
`inference.py` expects a JSON **list of objects**, not a single object.

## Acknowledgement

This implementation is inspired by the task framing and evaluation style of *Memes-as-Replies: Can Models Select Humorous Manga Panel Responses?* It is a practical PyTorch adaptation for this dataset and local training setup rather than a strict reproduction of the original paper.
