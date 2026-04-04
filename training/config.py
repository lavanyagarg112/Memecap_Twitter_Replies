from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field


@dataclass
class DataConfig:
    train_csv: str = "data/clean/train_clean.csv"
    val_csv: str = "data/clean/val_clean.csv"
    test_csv: str = "data/clean/test_clean.csv"
    min_candidates_per_task: int = 2
    candidate_text_fields: tuple[str, ...] = (
        "meme_title",
        "img_captions",
        "meme_captions",
        "metaphors",
    )
    use_image_features: bool = False


@dataclass
class TextConfig:
    max_vocab_size: int = 50000
    max_context_len: int = 64
    max_candidate_len: int = 96
    lowercase: bool = True
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"


@dataclass
class ModelConfig:
    model_type: str = "preference"
    text_encoder_type: str = "gru"
    embed_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.2
    use_shared_encoder: bool = True
    similarity_type: str = "cosine"


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    loss_type: str = "hinge"
    num_workers: int = 0
    save_dir: str = "checkpoints"
    use_amp: bool = False


@dataclass
class EvalConfig:
    ndcg_k: int = 5
    consensus_threshold: float = 0.8
    recall_target: str = "rank1"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    text: TextConfig = field(default_factory=TextConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_config() -> Config:
    return Config()


def parse_args() -> Config:
    config = get_default_config()
    parser = argparse.ArgumentParser(description="Train meme reply selection models.")

    parser.add_argument("--train_csv", type=str, default=config.data.train_csv)
    parser.add_argument("--val_csv", type=str, default=config.data.val_csv)
    parser.add_argument("--test_csv", type=str, default=config.data.test_csv)
    parser.add_argument("--min_candidates_per_task", type=int, default=config.data.min_candidates_per_task)
    parser.add_argument(
        "--candidate_text_fields",
        type=str,
        nargs="+",
        default=list(config.data.candidate_text_fields),
    )
    parser.add_argument("--use_image_features", action="store_true", default=config.data.use_image_features)

    parser.add_argument("--max_vocab_size", type=int, default=config.text.max_vocab_size)
    parser.add_argument("--max_context_len", type=int, default=config.text.max_context_len)
    parser.add_argument("--max_candidate_len", type=int, default=config.text.max_candidate_len)
    parser.add_argument("--lowercase", type=int, choices=[0, 1], default=int(config.text.lowercase))
    parser.add_argument("--unk_token", type=str, default=config.text.unk_token)
    parser.add_argument("--pad_token", type=str, default=config.text.pad_token)

    parser.add_argument("--model_type", type=str, choices=["similarity", "preference", "image", "multimodal"], default=config.model.model_type)
    parser.add_argument("--text_encoder_type", type=str, choices=["bow_mean", "gru"], default=config.model.text_encoder_type)
    parser.add_argument("--embed_dim", type=int, default=config.model.embed_dim)
    parser.add_argument("--hidden_dim", type=int, default=config.model.hidden_dim)
    parser.add_argument("--dropout", type=float, default=config.model.dropout)
    parser.add_argument("--use_shared_encoder", type=int, choices=[0, 1], default=int(config.model.use_shared_encoder))
    parser.add_argument("--similarity_type", type=str, choices=["cosine", "dot"], default=config.model.similarity_type)

    parser.add_argument("--seed", type=int, default=config.train.seed)
    parser.add_argument("--batch_size", type=int, default=config.train.batch_size)
    parser.add_argument("--num_epochs", type=int, default=config.train.num_epochs)
    parser.add_argument("--lr", type=float, default=config.train.lr)
    parser.add_argument("--weight_decay", type=float, default=config.train.weight_decay)
    parser.add_argument("--grad_clip_norm", type=float, default=config.train.grad_clip_norm)
    parser.add_argument("--device", type=str, default=config.train.device)
    parser.add_argument("--loss_type", type=str, choices=["hinge", "bpr"], default=config.train.loss_type)
    parser.add_argument("--num_workers", type=int, default=config.train.num_workers)
    parser.add_argument("--save_dir", type=str, default=config.train.save_dir)
    parser.add_argument("--use_amp", type=int, choices=[0, 1], default=int(config.train.use_amp))

    parser.add_argument("--ndcg_k", type=int, default=config.eval.ndcg_k)
    parser.add_argument("--consensus_threshold", type=float, default=config.eval.consensus_threshold)
    parser.add_argument("--recall_target", type=str, choices=["rank1", "avg_score"], default=config.eval.recall_target)

    args = parser.parse_args()

    config.data = DataConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        min_candidates_per_task=args.min_candidates_per_task,
        candidate_text_fields=tuple(args.candidate_text_fields),
        use_image_features=bool(args.use_image_features),
    )
    config.text = TextConfig(
        max_vocab_size=args.max_vocab_size,
        max_context_len=args.max_context_len,
        max_candidate_len=args.max_candidate_len,
        lowercase=bool(args.lowercase),
        unk_token=args.unk_token,
        pad_token=args.pad_token,
    )
    config.model = ModelConfig(
        model_type=args.model_type,
        text_encoder_type=args.text_encoder_type,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_shared_encoder=bool(args.use_shared_encoder),
        similarity_type=args.similarity_type,
    )
    config.train = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        loss_type=args.loss_type,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        use_amp=bool(args.use_amp),
    )
    config.eval = EvalConfig(
        ndcg_k=args.ndcg_k,
        consensus_threshold=args.consensus_threshold,
        recall_target=args.recall_target,
    )
    return config
