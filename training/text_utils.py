from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

import torch

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1


def _tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    text = re.sub(r"[^a-z0-9'\-|]", " ", text)
    return text.split()


class Vocab:
    def __init__(self, max_size: int = 20_000, lowercase: bool = True):
        self.max_size  = max_size
        self.lowercase = lowercase
        self._token2id: dict = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self._id2token: list = [PAD_TOKEN, UNK_TOKEN]

    def fit(self, texts: List[str]) -> "Vocab":
        counter: Counter = Counter()
        for text in texts:
            counter.update(_tokenize(text, self.lowercase))
        for token, _ in counter.most_common(self.max_size - 2):
            if token not in self._token2id:
                idx = len(self._id2token)
                self._token2id[token] = idx
                self._id2token.append(token)
        return self

    def encode(self, text: str, max_len: int) -> Tuple[List[int], List[int]]:
        tokens  = _tokenize(text, self.lowercase)[:max_len]
        ids     = [self._token2id.get(t, UNK_IDX) for t in tokens]
        mask    = [1] * len(ids)
        pad_len = max_len - len(ids)
        ids  += [PAD_IDX] * pad_len
        mask += [0]       * pad_len
        return ids, mask

    def encode_batch(
        self, texts: List[str], max_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_ids, all_mask = [], []
        for text in texts:
            ids, mask = self.encode(text, max_len)
            all_ids.append(ids)
            all_mask.append(mask)
        return (
            torch.tensor(all_ids,  dtype=torch.long),
            torch.tensor(all_mask, dtype=torch.long),
        )

    def state_dict(self) -> dict:
        return {
            "token2id":  self._token2id,
            "id2token":  self._id2token,
            "max_size":  self.max_size,
            "lowercase": self.lowercase,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "Vocab":
        v = cls(max_size=d["max_size"], lowercase=d["lowercase"])
        v._token2id = d["token2id"]
        v._id2token = d["id2token"]
        return v

    def __len__(self) -> int:
        return len(self._id2token)


def build_vocab(texts: List[str], max_size: int = 20_000) -> Vocab:
    v = Vocab(max_size=max_size)
    v.fit(texts)
    print(f"[Vocab] {len(v)} tokens from {len(texts)} strings.")
    return v
