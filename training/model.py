from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import requests



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

class QwenImageRanker(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )

        hidden_dim = config.model.hidden_dim

        self.scorer = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def load_image(self, url):
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(response.raw).convert("RGB")
            return img
        except:
            return Image.new("RGB", (224, 224), color="white")

    def forward(self, batch: Batch) -> torch.Tensor:
        batch_size = len(batch.image_urls)
        num_candidates = len(batch.image_urls[0])

        all_scores = []

        for i in range(batch_size):
            tweet = batch.context_texts[i]

            meme_scores = []

            for j in range(num_candidates):
                url = batch.image_urls[i][j]

                image = self.load_image(url)

                prompt = f"Tweet: {tweet}\nHow well does this meme reply?"

                inputs = self.processor(
                    text=[prompt],
                    images=[image],
                    return_tensors="pt"
                ).to(self.model.device)

                outputs = self.model(**inputs, output_hidden_states=True)

                embedding = outputs.hidden_states[-1][:, 0, :]

                score = self.scorer(embedding).squeeze(-1)

                meme_scores.append(score)

            meme_scores = torch.stack(meme_scores)
            all_scores.append(meme_scores)

        scores = torch.stack(all_scores)

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

class MultimodalRanker(nn.Module):
    def __init__(self, config, vocab_size, pad_idx):
        super().__init__()

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            text_encoder_type=config.model.text_encoder_type,
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )

        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )

        combined_dim = self.text_encoder.output_dim + self.qwen.config.hidden_size

        self.scorer = nn.Sequential(
            nn.Linear(combined_dim, config.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.model.hidden_dim, 1)
        )

    def load_image(self, url):
        try:
            response = requests.get(url, timeout=5)
            return Image.open(response.raw).convert("RGB")
        except:
            return Image.new("RGB", (224, 224), color="white")

    def forward(self, batch: Batch) -> torch.Tensor:
        batch_size, num_candidates, candidate_len = batch.candidate_input_ids.shape

        context_emb = self.text_encoder(
            batch.context_input_ids,
            batch.context_attention_mask
        )

        flat_ids = batch.candidate_input_ids.view(batch_size * num_candidates, candidate_len)
        flat_mask = batch.candidate_attention_mask.view(batch_size * num_candidates, candidate_len)

        text_emb = self.text_encoder(flat_ids, flat_mask)
        text_emb = text_emb.view(batch_size, num_candidates, -1)

        flat_urls = []
        for i in range(batch_size):
            for j in range(num_candidates):
                flat_urls.append(batch.metadata[i][j].get("image_url", ""))

        images = [self.load_image(url) for url in flat_urls]

        inputs = self.processor(images=images, return_tensors="pt").to(self.qwen.device)
        outputs = self.qwen(**inputs, output_hidden_states=True)

        image_emb = outputs.hidden_states[-1][:, 0, :]
        image_emb = image_emb.view(batch_size, num_candidates, -1)

        expanded_context = context_emb.unsqueeze(1).expand_as(text_emb)

        combined = torch.cat(
            [expanded_context, text_emb + image_emb],
            dim=-1
        )

        scores = self.scorer(combined).squeeze(-1)

        return scores.masked_fill(batch.candidate_mask <= 0, -1e9)


def build_model(config: Config, vocab_size: int) -> nn.Module:
    pad_idx = 0
    if config.model.model_type == "similarity":
        return SimilarityRanker(config, vocab_size=vocab_size, pad_idx=pad_idx)
    if config.model.model_type == "image":
        return QwenImageRanker(config)
    if config.model.model_type == "preference":
        return PreferenceRanker(config, vocab_size=vocab_size, pad_idx=pad_idx)
    if config.model.model_type == "multimodal":
        return MultimodalRanker(config, vocab_size=vocab_size, pad_idx=pad_idx)
    raise ValueError(f"Unsupported model_type: {config.model.model_type}")
