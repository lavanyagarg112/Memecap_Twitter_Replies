"""
3 pipelines are supported via --pipeline:
  text        tweet text  +  candidate text                   (HF / BOW / GRU)
  image       tweet text  +  candidate image                  (Qwen2.5-VL)
  multimodal  tweet text  +  candidate image + candidate text (Qwen2.5-VL)

4 backends are exposed via --encoder_type:
  hf        any HuggingFace sentence-transformer / BERT-style model
  bow_mean  bag-of-words mean-pooling baseline (no external download)
  gru       GRU encoder baseline               (no external download)
  qwen_vl   Qwen2.5-VL cross-encoder           (image / multimodal pipelines)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

_TRAINING_DIR = Path(__file__).resolve().parent
_NON_ANNOTATION_DIR = _TRAINING_DIR / "data" / "non-annotation-dataset" / "clean"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    train_csv: str = str(_NON_ANNOTATION_DIR / "train_clean.csv")
    val_csv:   str = str(_NON_ANNOTATION_DIR / "val_clean.csv")
    test_csv:  str = str(_NON_ANNOTATION_DIR / "test_clean.csv")

    # Folder containing pre-downloaded images named <meme_post_id>.jpg/png.
    # Set to "" to fall back to downloading from image_url at runtime.
    image_dir: str = "data/images"

    # Text fields joined to form each candidate's text representation.
    candidate_text_fields: List[str] = field(
        default_factory=lambda: ["meme_title", "img_captions", "meme_captions", "metaphors"]
    )

    min_candidates_per_task: int = 2
    max_candidates_per_task: int = 0   # kept for checkpoint compatibility; runtime truncation is disabled


@dataclass
class TextConfig:
    max_context_len: int  = 128
    max_cand_len:    int  = 128
    lowercase:       bool = True


@dataclass
class ModelConfig:
    pipeline: str = "text"        # "text" | "image" | "multimodal"

    # "hf"       --> HuggingFace AutoModel (text pipelines)
    # "bow_mean" --> bag-of-words baseline
    # "gru"      --> GRU baseline
    # "qwen_vl"  --> Qwen2.5-VL cross-encoder (image / multimodal pipelines)
    encoder_type: str = "hf"

    hf_model_name:    str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model_name:  str = "openai/clip-vit-base-patch32"
    qwen_vl_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    llava_model_name: str = "llava-hf/llava-1.5-7b-hf"

    ranker_type:   str = "preference"   # "similarity" | "preference"
    similarity_fn: str = "cosine"       # "cosine" | "dot"

    shared_encoder: bool = True
    proj_dim:       int  = 256   # 0 = skip projection
    mlp_hidden:     int  = 512
    dropout:        float = 0.1
    freeze_encoder: bool  = False

    # BOW / GRU specific
    vocab_size: int = 20_000
    embed_dim:  int = 128
    hidden_dim: int = 256


@dataclass
class TrainConfig:
    num_epochs:     int   = 10
    batch_size:     int   = 16
    lr:             float = 2e-5
    weight_decay:   float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_steps:   int   = 100
    loss_type:    str   = "bpr"   # "bpr" | "hinge"
    hinge_margin: float = 1.0
    device:      str  = "cuda"
    seed:        int  = 42
    num_workers: int  = 2
    use_amp:     bool = True      # mixed precision, CUDA only
    save_dir:  str = "checkpoints"
    resume_from: str = ""
    log_every: int = 50


@dataclass
class EvalConfig:
    ndcg_k:        int = 10
    recall_target: str = "rank1"   # only "rank1" for now (no avg_score column)


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    text:  TextConfig  = field(default_factory=TextConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval:  EvalConfig  = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            data  = DataConfig(**d["data"]),
            model = ModelConfig(**d["model"]),
            text  = TextConfig(**d["text"]),
            train = TrainConfig(**d["train"]),
            eval  = EvalConfig(**d["eval"]),
        )


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Train a meme reply ranker (text / image / multimodal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--train_csv",      default=cfg.data.train_csv)
    parser.add_argument("--val_csv",        default=cfg.data.val_csv)
    parser.add_argument("--test_csv",       default=cfg.data.test_csv)
    parser.add_argument("--image_dir",      default=cfg.data.image_dir,
                   help="Folder with pre-downloaded images. "
                        "Pass '' to download from image_url at runtime.")
    parser.add_argument("--min_candidates", type=int, default=cfg.data.min_candidates_per_task)

    parser.add_argument("--pipeline",      default=cfg.model.pipeline,
                   choices=["text", "image", "multimodal"])
    parser.add_argument("--encoder_type",  default=cfg.model.encoder_type,
                   choices=["hf", "bow_mean", "gru", "qwen_vl"])
    parser.add_argument("--hf_model",      default=cfg.model.hf_model_name)
    parser.add_argument("--qwen_vl_model", default=cfg.model.qwen_vl_model_name)
    parser.add_argument("--ranker_type",   default=cfg.model.ranker_type,
                   choices=["similarity", "preference"])
    parser.add_argument("--proj_dim",      type=int, default=cfg.model.proj_dim)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--no_shared_encoder", action="store_true",
                   help="Use separate encoders for context and candidates.")

    parser.add_argument("--max_context_len", type=int, default=cfg.text.max_context_len)
    parser.add_argument("--max_cand_len",    type=int, default=cfg.text.max_cand_len)

    parser.add_argument("--num_epochs",    type=int,   default=cfg.train.num_epochs)
    parser.add_argument("--batch_size",    type=int,   default=cfg.train.batch_size)
    parser.add_argument("--lr",            type=float, default=cfg.train.lr)
    parser.add_argument("--weight_decay",  type=float, default=cfg.train.weight_decay)
    parser.add_argument("--loss_type",     default=cfg.train.loss_type,
                   choices=["bpr", "hinge"])
    parser.add_argument("--hinge_margin",  type=float, default=cfg.train.hinge_margin)
    parser.add_argument("--device",        default=cfg.train.device)
    parser.add_argument("--seed",          type=int,   default=cfg.train.seed)
    parser.add_argument("--save_dir",      default=cfg.train.save_dir)
    parser.add_argument("--resume",        nargs="?", const="auto", default=cfg.train.resume_from,
                   help="Resume from a checkpoint path. Pass without a value to use <save_dir>/latest.pt.")
    parser.add_argument("--num_workers",   type=int,   default=cfg.train.num_workers)
    parser.add_argument("--no_amp",        action="store_true",
                   help="Disable mixed-precision training.")

    parser.add_argument("--ndcg_k", type=int, default=cfg.eval.ndcg_k)
    args = parser.parse_args()

    cfg.data.train_csv               = args.train_csv
    cfg.data.val_csv                 = args.val_csv
    cfg.data.test_csv                = args.test_csv
    cfg.data.image_dir               = args.image_dir
    cfg.data.min_candidates_per_task = args.min_candidates
    cfg.data.max_candidates_per_task = 0

    cfg.model.pipeline         = args.pipeline
    cfg.model.encoder_type     = args.encoder_type
    cfg.model.hf_model_name    = args.hf_model
    cfg.model.qwen_vl_model_name = args.qwen_vl_model
    cfg.model.ranker_type      = args.ranker_type
    cfg.model.proj_dim         = args.proj_dim
    cfg.model.freeze_encoder   = args.freeze_encoder
    cfg.model.shared_encoder   = not args.no_shared_encoder

    cfg.text.max_context_len = args.max_context_len
    cfg.text.max_cand_len    = args.max_cand_len

    cfg.train.num_epochs   = args.num_epochs
    cfg.train.batch_size   = args.batch_size
    cfg.train.lr           = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.loss_type    = args.loss_type
    cfg.train.hinge_margin = args.hinge_margin
    cfg.train.device       = args.device
    cfg.train.seed         = args.seed
    cfg.train.save_dir     = args.save_dir
    cfg.train.resume_from  = args.resume
    cfg.train.num_workers  = args.num_workers
    cfg.train.use_amp      = not args.no_amp

    cfg.eval.ndcg_k = args.ndcg_k

    valid_text_encoders = {"hf", "bow_mean", "gru"}
    if cfg.model.pipeline == "text":
        if cfg.model.encoder_type not in valid_text_encoders:
            parser.error("--pipeline text only supports --encoder_type in {hf, bow_mean, gru}.")
    else:
        if cfg.model.encoder_type != "qwen_vl":
            parser.error(f"--pipeline {cfg.model.pipeline} requires --encoder_type qwen_vl.")

    if cfg.model.encoder_type == "qwen_vl" and cfg.model.ranker_type != "preference":
        parser.error("--encoder_type qwen_vl requires --ranker_type preference.")

    return cfg
