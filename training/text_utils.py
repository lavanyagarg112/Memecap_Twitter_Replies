from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str, lowercase: bool = True) -> str:
    if text is None:
        text = ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def safe_join_text(parts: list[str]) -> str:
    cleaned = []
    for part in parts:
        if part is None:
            continue
        value = str(part).strip()
        if not value or value.lower() == "nan":
            continue
        cleaned.append(value)
    return " [SEP] ".join(cleaned)


def build_candidate_text(row: dict, text_fields: tuple[str, ...]) -> str:
    return safe_join_text([str(row.get(field, "") or "") for field in text_fields])


def tokenize(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return text.split()


class Vocab:
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>") -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []
        self._add_special(self.pad_token)
        self._add_special(self.unk_token)

    def _add_special(self, token: str) -> None:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    def fit(self, texts: list[str], max_size: int | None = None) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        max_tokens = None if max_size is None else max(max_size - len(self.itos), 0)
        for token, _ in counter.most_common(max_tokens):
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        decoded = []
        for idx in ids:
            if 0 <= idx < len(self.itos):
                decoded.append(self.itos[idx])
            else:
                decoded.append(self.unk_token)
        return decoded

    def __len__(self) -> int:
        return len(self.itos)

    def state_dict(self) -> dict:
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "itos": self.itos,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "Vocab":
        vocab = cls(
            pad_token=state["pad_token"],
            unk_token=state["unk_token"],
        )
        vocab.itos = list(state["itos"])
        vocab.stoi = {token: idx for idx, token in enumerate(vocab.itos)}
        return vocab


def encode_text(text: str, vocab: Vocab, max_len: int) -> tuple[list[int], list[int]]:
    tokens = tokenize(text)[:max_len]
    token_ids = vocab.encode(tokens)
    attention_mask = [1] * len(token_ids)
    return token_ids, attention_mask
