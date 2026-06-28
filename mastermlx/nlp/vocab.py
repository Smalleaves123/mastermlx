from __future__ import annotations

import numpy as np

from .tok import SimpleTokenizer


class Vocab:
    """Build a token-to-id mapping from text."""

    def __init__(self, tokenizer=None, min_freq=1, max_size=None, pad="<pad>", unk="<unk>"):
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.min_freq = int(min_freq)
        self.max_size = None if max_size is None else int(max_size)
        self.pad = pad
        self.unk = unk
        self.stoi_ = {}
        self.itos_ = []
        self.fitted_ = False

    def fit(self, texts):
        counts = {}
        for text in texts:
            for tok in self.tokenizer(text):
                counts[tok] = counts.get(tok, 0) + 1

        items = [(tok, n) for tok, n in counts.items() if n >= self.min_freq]
        items.sort(key=lambda item: (-item[1], item[0]))
        if self.max_size is not None:
            items = items[: max(0, self.max_size - 2)]

        self.itos_ = [self.pad, self.unk] + [tok for tok, _ in items if tok not in {self.pad, self.unk}]
        self.stoi_ = {tok: i for i, tok in enumerate(self.itos_)}
        self.fitted_ = True
        return self

    def __len__(self):
        return len(self.itos_)

    def encode(self, text):
        if not self.fitted_:
            raise RuntimeError("Vocab has not been fit yet")
        return np.asarray([self.stoi_.get(tok, self.stoi_[self.unk]) for tok in self.tokenizer(text)], dtype=int)

    def decode(self, ids, drop_pad=True):
        if not self.fitted_:
            raise RuntimeError("Vocab has not been fit yet")
        ids = np.asarray(ids, dtype=int).ravel()
        out = []
        for idx in ids:
            tok = self.itos_[idx] if 0 <= idx < len(self.itos_) else self.unk
            if drop_pad and tok == self.pad:
                continue
            out.append(tok)
        return out


class SeqPad:
    """Pad or truncate token id sequences to a fixed width."""

    def __init__(self, max_len, pad_id=0, trunc="right"):
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.trunc = trunc

    def __call__(self, seq):
        if not isinstance(seq, list):
            seq = list(seq)
        if len(seq) > self.max_len:
            if self.trunc == "left":
                seq = seq[-self.max_len :]
            elif self.trunc == "right":
                seq = seq[: self.max_len]
            else:
                raise ValueError("trunc must be one of: left, right")
        out = np.full(self.max_len, self.pad_id, dtype=int)
        out[: len(seq)] = seq
        return out


class TextSeq:
    """Turn raw text into padded integer token sequences."""

    def __init__(self, tokenizer=None, min_freq=1, max_size=None, max_len=None, pad="<pad>", unk="<unk>"):
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.min_freq = int(min_freq)
        self.max_size = max_size
        self.max_len = None if max_len is None else int(max_len)
        self.pad = pad
        self.unk = unk
        self.vocab_ = None
        self.pad_ = None

    def fit(self, texts, y=None):
        self.vocab_ = Vocab(self.tokenizer, min_freq=self.min_freq, max_size=self.max_size, pad=self.pad, unk=self.unk).fit(texts)
        self.pad_ = SeqPad(self.max_len or 0, pad_id=self.vocab_.stoi_[self.pad]) if self.max_len else None
        return self

    def transform(self, texts):
        if self.vocab_ is None:
            raise RuntimeError("TextSeq has not been fit yet")
        seqs = [self.vocab_.encode(text) for text in texts]
        if not seqs:
            width = self.max_len if self.max_len is not None else 0
            return np.zeros((0, width), dtype=int)
        if self.max_len is None:
            max_len = max((len(seq) for seq in seqs), default=0)
            pad = SeqPad(max_len, pad_id=self.vocab_.stoi_[self.pad])
        else:
            pad = self.pad_
        return np.stack([pad(seq) for seq in seqs], axis=0)

    def fit_transform(self, texts, y=None):
        return self.fit(texts, y).transform(texts)
