from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from dataset import Batch


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, text_encoder_type: str, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        if text_encoder_type == "gru":
            self.gru = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.output_dim = hidden_dim
        elif text_encoder_type == "bow_mean":
            self.output_dim = embed_dim
        else:
            raise ValueError(f"Unsupported text_encoder_type: {text_encoder_type}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(self.embedding(input_ids))
        attention_mask = attention_mask.float()

        if self.text_encoder_type == "bow_mean":
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return masked_embeddings.sum(dim=1) / lengths

        lengths = attention_mask.sum(dim=1).long().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]


class SimilarityRanker(nn.Module):
    def __init__(self, config: Config, vocab_size: int, pad_idx: int) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = TextEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            text_encoder_type=config.model.text_encoder_type,
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )
        if config.model.use_shared_encoder:
            self.candidate_encoder = self.context_encoder
        else:
            self.candidate_encoder = TextEncoder(
                vocab_size=vocab_size,
                pad_idx=pad_idx,
                text_encoder_type=config.model.text_encoder_type,
                embed_dim=config.model.embed_dim,
                hidden_dim=config.model.hidden_dim,
                dropout=config.model.dropout,
            )

    def forward(self, batch: Batch) -> torch.Tensor:
        batch_size, num_candidates, candidate_len = batch.candidate_input_ids.shape
        context_embeddings = self.context_encoder(batch.context_input_ids, batch.context_attention_mask)
        flat_candidate_ids = batch.candidate_input_ids.view(batch_size * num_candidates, candidate_len)
        flat_candidate_mask = batch.candidate_attention_mask.view(batch_size * num_candidates, candidate_len)
        candidate_embeddings = self.candidate_encoder(flat_candidate_ids, flat_candidate_mask)
        candidate_embeddings = candidate_embeddings.view(batch_size, num_candidates, -1)

        if self.config.model.similarity_type == "cosine":
            context_embeddings = F.normalize(context_embeddings, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
            scores = (candidate_embeddings * context_embeddings.unsqueeze(1)).sum(dim=-1)
        elif self.config.model.similarity_type == "dot":
            scores = (candidate_embeddings * context_embeddings.unsqueeze(1)).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported similarity_type: {self.config.model.similarity_type}")

        return scores.masked_fill(batch.candidate_mask <= 0, -1e9)


class PreferenceRanker(nn.Module):
    def __init__(self, config: Config, vocab_size: int, pad_idx: int) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = TextEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            text_encoder_type=config.model.text_encoder_type,
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )
        if config.model.use_shared_encoder:
            self.candidate_encoder = self.context_encoder
        else:
            self.candidate_encoder = TextEncoder(
                vocab_size=vocab_size,
                pad_idx=pad_idx,
                text_encoder_type=config.model.text_encoder_type,
                embed_dim=config.model.embed_dim,
                hidden_dim=config.model.hidden_dim,
                dropout=config.model.dropout,
            )

        feature_dim = self.context_encoder.output_dim * 4
        hidden_dim = config.model.hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        batch_size, num_candidates, candidate_len = batch.candidate_input_ids.shape
        context_embeddings = self.context_encoder(batch.context_input_ids, batch.context_attention_mask)
        flat_candidate_ids = batch.candidate_input_ids.view(batch_size * num_candidates, candidate_len)
        flat_candidate_mask = batch.candidate_attention_mask.view(batch_size * num_candidates, candidate_len)
        candidate_embeddings = self.candidate_encoder(flat_candidate_ids, flat_candidate_mask)
        candidate_embeddings = candidate_embeddings.view(batch_size, num_candidates, -1)

        expanded_context = context_embeddings.unsqueeze(1).expand_as(candidate_embeddings)
        features = torch.cat(
            [
                expanded_context,
                candidate_embeddings,
                torch.abs(expanded_context - candidate_embeddings),
                expanded_context * candidate_embeddings,
            ],
            dim=-1,
        )
        scores = self.scorer(features).squeeze(-1)
        return scores.masked_fill(batch.candidate_mask <= 0, -1e9)


def build_model(config: Config, vocab_size: int) -> nn.Module:
    pad_idx = 0
    if config.model.model_type == "similarity":
        return SimilarityRanker(config, vocab_size=vocab_size, pad_idx=pad_idx)
    if config.model.model_type == "preference":
        return PreferenceRanker(config, vocab_size=vocab_size, pad_idx=pad_idx)
    raise ValueError(f"Unsupported model_type: {config.model.model_type}")
