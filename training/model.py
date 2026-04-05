from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch


class _Proj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _HFTextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = False):
        super().__init__()
        from transformers import AutoModel
        self.model   = AutoModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out  = self.model(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class _CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = False):
        super().__init__()
        from transformers import CLIPModel
        self.clip    = CLIPModel.from_pretrained(model_name)
        self.out_dim = self.clip.config.text_config.hidden_size
        if freeze:
            for p in self.clip.text_model.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)


class _BOWEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.out_dim = embed_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb  = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class _GRUEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru     = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out_dim = hidden_dim * 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb     = self.embed(input_ids)
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)   # h: [2, N, H]
        return torch.cat([h[0], h[1]], dim=-1)


class _CLIPImageEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = False):
        super().__init__()
        from transformers import CLIPModel
        self.clip    = CLIPModel.from_pretrained(model_name)
        self.out_dim = self.clip.config.vision_config.hidden_size
        if freeze:
            for p in self.clip.vision_model.parameters():
                p.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip.get_image_features(pixel_values=pixel_values)


class _LLaVAEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = False):
        super().__init__()
        from transformers import LlavaForConditionalGeneration
        self.model   = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype = torch.float16,
            low_cpu_mem_usage = True,
        )
        self.out_dim = self.model.config.text_config.hidden_size
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values:   torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            pixel_values   = pixel_values,
            output_hidden_states = True,
        )
        last_hidden = out.hidden_states[-1]    # [N, L, D]
        seq_lens = attention_mask.sum(1) - 1   # [N]
        idx      = seq_lens.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(
            -1, 1, last_hidden.size(-1)
        )
        emb = last_hidden.gather(1, idx).squeeze(1)   # [N, D]
        return emb.float()


class _SimilarityRanker(nn.Module):
    def __init__(self, ctx_dim: int, cand_dim: int, proj_dim: int,
                 sim_fn: str = "cosine", dropout: float = 0.1):
        super().__init__()
        self.ctx_proj  = _Proj(ctx_dim,  proj_dim, dropout)
        self.cand_proj = _Proj(cand_dim, proj_dim, dropout)
        self.sim_fn    = sim_fn

    def forward(self, ctx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """ctx [B,D], cand [B,K,D] → scores [B,K]"""
        c = self.ctx_proj(ctx).unsqueeze(1).expand_as(
            self.cand_proj(cand)
        )
        m = self.cand_proj(cand)
        if self.sim_fn == "cosine":
            return F.cosine_similarity(c, m, dim=-1)
        return (c * m).sum(-1)


class _PreferenceRanker(nn.Module):
    def __init__(self, ctx_dim: int, cand_dim: int, proj_dim: int,
                 mlp_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ctx_proj  = _Proj(ctx_dim,  proj_dim, dropout)
        self.cand_proj = _Proj(cand_dim, proj_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 4, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, ctx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """ctx [B,D], cand [B,K,D] → scores [B,K]"""
        c = self.ctx_proj(ctx).unsqueeze(1).expand(
            ctx.shape[0], cand.shape[1], -1
        )
        m = self.cand_proj(cand)
        f = torch.cat([c, m, (c - m).abs(), c * m], dim=-1)
        return self.mlp(f).squeeze(-1)


class MemeRanker(nn.Module):
    def __init__(self, cfg, vocab_size: int = 2):
        super().__init__()
        mcfg     = cfg.model
        pipeline = mcfg.pipeline
        enc_type = mcfg.encoder_type
        freeze   = mcfg.freeze_encoder

        self.pipeline  = pipeline
        self.enc_type  = enc_type
        self.is_llava  = (enc_type == "llava")

        self.ctx_enc = self._make_text_enc(mcfg, vocab_size, freeze)

        if enc_type == "llava":
            self.llava_enc = _LLaVAEncoder(mcfg.llava_model_name, freeze=freeze)
            cand_dim = self.llava_enc.out_dim
            self.cand_text_enc = None
            self.cand_img_enc  = None

        elif pipeline == "text":
            self.cand_text_enc = (
                self.ctx_enc if mcfg.shared_encoder
                else self._make_text_enc(mcfg, vocab_size, freeze)
            )
            self.cand_img_enc  = None
            cand_dim = self.cand_text_enc.out_dim

        elif pipeline == "image":
            self.cand_img_enc  = _CLIPImageEncoder(mcfg.clip_model_name, freeze)
            self.cand_text_enc = None
            cand_dim = self.cand_img_enc.out_dim

        else:  # multimodal (non-LLaVA): fuse image + text
            self.cand_img_enc  = _CLIPImageEncoder(mcfg.clip_model_name, freeze)
            self.cand_text_enc = (
                self.ctx_enc if mcfg.shared_encoder
                else self._make_text_enc(mcfg, vocab_size, freeze)
            )
            fuse_in  = self.cand_img_enc.out_dim + self.cand_text_enc.out_dim
            proj_dim = mcfg.proj_dim if mcfg.proj_dim > 0 else fuse_in
            self.fusion = nn.Linear(fuse_in, proj_dim)
            cand_dim = proj_dim

        ctx_dim  = self.ctx_enc.out_dim
        proj_dim = mcfg.proj_dim if mcfg.proj_dim > 0 else max(ctx_dim, cand_dim)

        if mcfg.ranker_type == "similarity":
            self.ranker = _SimilarityRanker(
                ctx_dim, cand_dim, proj_dim,
                sim_fn  = mcfg.similarity_fn,
                dropout = mcfg.dropout,
            )
        else:
            self.ranker = _PreferenceRanker(
                ctx_dim, cand_dim, proj_dim,
                mlp_hidden = mcfg.mlp_hidden,
                dropout    = mcfg.dropout,
            )


    @staticmethod
    def _make_text_enc(mcfg, vocab_size: int, freeze: bool) -> nn.Module:
        enc = mcfg.encoder_type
        if enc == "hf":
            return _HFTextEncoder(mcfg.hf_model_name, freeze)
        elif enc == "clip":
            return _CLIPTextEncoder(mcfg.clip_model_name, freeze)
        elif enc == "bow_mean":
            return _BOWEncoder(vocab_size, mcfg.embed_dim)
        elif enc == "gru":
            return _GRUEncoder(vocab_size, mcfg.embed_dim, mcfg.hidden_dim)
        elif enc == "llava":
            return _CLIPTextEncoder(mcfg.clip_model_name, freeze)
        raise ValueError(f"Unknown encoder_type: {enc!r}")


    def _encode_context(self, batch: Batch) -> torch.Tensor:
        return self.ctx_enc(batch.context_input_ids, batch.context_attention_mask)

    def _encode_candidates(self, batch: Batch) -> torch.Tensor:
        B, K = batch.ranks.shape

        if self.is_llava:
            flat_ids  = batch.candidate_input_ids.view(B * K, -1)
            flat_mask = batch.candidate_attention_mask.view(B * K, -1)
            flat_pv   = batch.pixel_values.view(B * K, *batch.pixel_values.shape[2:])
            emb = self.llava_enc(flat_ids, flat_mask, flat_pv)   # [B*K, D]
            return emb.view(B, K, -1)

        if self.pipeline == "text":
            flat_ids  = batch.candidate_input_ids.view(B * K, -1)
            flat_mask = batch.candidate_attention_mask.view(B * K, -1)
            emb = self.cand_text_enc(flat_ids, flat_mask)
            return emb.view(B, K, -1)

        if self.pipeline == "image":
            flat_pv = batch.pixel_values.view(B * K, *batch.pixel_values.shape[2:])
            emb = self.cand_img_enc(flat_pv)
            return emb.view(B, K, -1)

        # multimodal (non-LLaVA): fuse
        flat_ids  = batch.candidate_input_ids.view(B * K, -1)
        flat_mask = batch.candidate_attention_mask.view(B * K, -1)
        flat_pv   = batch.pixel_values.view(B * K, *batch.pixel_values.shape[2:])
        text_emb  = self.cand_text_enc(flat_ids, flat_mask)
        img_emb   = self.cand_img_enc(flat_pv)
        fused     = self.fusion(torch.cat([text_emb, img_emb], dim=-1))
        return fused.view(B, K, -1)

    def forward(self, batch: Batch) -> torch.Tensor:
        ctx  = self._encode_context(batch)     # [B, D]
        cand = self._encode_candidates(batch)  # [B, K, D']
        return self.ranker(ctx, cand)          # [B, K]


def build_model(config, vocab_size: int = 2) -> MemeRanker:
    model  = MemeRanker(config, vocab_size=vocab_size)
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Model] pipeline={config.model.pipeline}  "
        f"encoder={config.model.encoder_type}  "
        f"ranker={config.model.ranker_type}  "
        f"params={total:,}  trainable={trainable:,}"
    )
    return model
