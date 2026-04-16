# Meme Reply Selection

This project studies meme reply selection as a ranking task. Given a tweet and a fixed set of candidate memes, the goal is to rank the most suitable meme reply at the top.

The repository contains:

- data curation pipelines for annotation-based and non-annotation-based training data
- reranker training code for text, image, and multimodal models
- evaluation scripts and supporting analysis utilities

## Repository Structure

```text
project/
├── pre-training/               # Data curation pipelines
│   ├── annotation_app/         # Flask app for human annotation
│   ├── non-annotation/         # Real tweet pipeline with VLM-based reranking
│   ├── annotate_parallel.py    # Multi-model VLM annotation
│   ├── create_train_data.py    # Train/val/test split creation
│   └── README.md
├── training/                   # Reranker training and evaluation
│   ├── data/
│   ├── download_images.py
│   ├── train.py
│   ├── eval.py
│   └── README.md
└── README.md
```

## Data Pipelines

### 1. Non-annotation pipeline

This is the primary dataset curation pipeline.

- starts from real tweet-meme pairs from `HSDSLab/TwitterMemes`
- uses MemeCap as the candidate meme bank
- uses VLM descriptions plus `all-MiniLM-L6-v2` similarity to build a candidate pool
- uses VLM selection and reranking to produce the final candidate set
- does not require human annotation

Main entry point:

```bash
cd pre-training/non-annotation
python rank_similar_memes.py --limit 500
```

### 2. Annotation pipeline

This pipeline creates synthetic tweet contexts, selects candidates, and collects human plus model annotations.

Main stages:

```text
Generate tweets -> Clean -> Filter -> Select candidates -> Human annotation -> Model annotation -> Rankings -> Train data
```

Main entry point:

```bash
cd pre-training
python create_train_data.py
```

More detailed instructions are in [pre-training/README.md](/Users/lavanya/Documents/Year3Sem2/CS4248/project/pre-training/README.md).

## Training Setup

Training is done as reranking over fixed candidate sets already stored in CSV files. The code groups rows by `task_id`, scores the candidates within each task, and learns to place better memes above worse ones.

The three training pipelines are:

1. `text`
   Uses tweet text and candidate meme text.
2. `image`
   Uses tweet text and candidate meme image.
3. `multimodal`
   Uses tweet text, candidate meme image, and candidate meme text.

The text pipeline fine-tunes a text encoder. The image and multimodal pipelines use `Qwen/Qwen2.5-VL-3B-Instruct` with a frozen VLM backbone and a trainable ranking head.

## Quick Start

Install dependencies and prepare images:

```bash
pip install -r training/requirements.txt
python training/download_images.py
```

Run training:

```bash
python training/train.py --pipeline text --encoder_type hf --device cuda --batch_size 16 --save_dir training/checkpoints/text_hf_clean
python training/train.py --pipeline image --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/image_qwen_clean
python training/train.py --pipeline multimodal --encoder_type qwen_vl --device cuda --batch_size 1 --freeze_encoder --image_dir training/data/non-annotation-dataset/images --save_dir training/checkpoints/multimodal_qwen_clean
```

Run evaluation:

```bash
python training/eval.py --checkpoint training/checkpoints/text_hf_clean/best.pt --split test --device cuda
```

More detailed training instructions are in [training/README.md](/Users/lavanya/Documents/Year3Sem2/CS4248/project/training/README.md).

## Data Format

Each row in the training CSVs corresponds to one tweet-meme candidate pair. Training groups rows by `task_id`.

Common columns include:

- `tweet_text`
- `meme_title`
- `img_captions`
- `meme_captions`
- `metaphors`
- `rank`

Here, `rank = 1` means the best meme within that task's candidate set.
