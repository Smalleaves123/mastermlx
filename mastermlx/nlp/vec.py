from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections import Counter

import numpy as np

from .tok import SimpleTokenizer, word_ngrams


@dataclass(frozen=True)
class SparseCOO:
    """Lightweight COO-style sparse matrix without a SciPy dependency."""

    row: np.ndarray
    col: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    def toarray(self):
        X = np.zeros(self.shape, dtype=float)
        if self.data.size:
            np.add.at(X, (self.row, self.col), self.data)
        return X

    def __array__(self, dtype=None):
        X = self.toarray()
        if dtype is not None:
            return X.astype(dtype, copy=False)
        return X


__all__ = ["SparseCOO", "CountVectorizer", "TfidfVectorizer", "HashingVectorizer"]


class CountVectorizer:
    """Bag-of-ngrams vectorizer backed by NumPy."""

    def __init__(self, tokenizer=None, ngram_range=(1, 1), lowercase=True, max_features=None, stop_words=None):
        self.tokenizer = tokenizer or SimpleTokenizer(lowercase=False)
        self.ngram_range = tuple(ngram_range)
        self.lowercase = bool(lowercase)
        self.max_features = max_features
        self.stop_words = stop_words
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.stop_words_ = None

    def _prepare_stop_words(self):
        if self.stop_words is None:
            return None
        if isinstance(self.stop_words, str):
            stop_words = {self.stop_words}
        else:
            stop_words = set(self.stop_words)
        if self.lowercase:
            stop_words = {word.lower() if isinstance(word, str) else word for word in stop_words}
        return stop_words

    def _tokens(self, text):
        if self.lowercase and isinstance(text, str):
            text = text.lower()
        return self.tokenizer(text)

    def _feats(self, text, stop_words=None):
        tokens = self._tokens(text)
        if stop_words:
            tokens = [tok for tok in tokens if tok not in stop_words]
        feats = []
        lo, hi = self.ngram_range
        for n in range(int(lo), int(hi) + 1):
            if n == 1:
                feats.extend(tokens)
            else:
                feats.extend([" ".join(ng) for ng in word_ngrams(tokens, n=n)])
        return feats

    def fit(self, texts, y=None):
        texts = list(texts)
        counts = Counter()
        self.stop_words_ = self._prepare_stop_words()
        for text in texts:
            counts.update(self._feats(text, stop_words=self.stop_words_))
        self._build_vocab(counts)
        return self

    def _build_vocab(self, counts):
        items = counts.most_common()
        if self.max_features is not None:
            items = items[: int(self.max_features)]
        self.feature_names_ = [tok for tok, _ in items]
        self.vocabulary_ = {tok: i for i, tok in enumerate(self.feature_names_)}

    def transform(self, texts, dense=True):
        if not self.vocabulary_:
            raise RuntimeError("Vectorizer has not been fit yet")
        texts = list(texts)
        n_rows = len(texts)
        n_cols = len(self.vocabulary_)
        stop_words = self.stop_words_
        if dense:
            X = np.zeros((n_rows, n_cols), dtype=float)
        else:
            rows = []
            cols = []
            data = []
        for i, text in enumerate(texts):
            counts = Counter(self._feats(text, stop_words=stop_words))
            for token, value in counts.items():
                idx = self.vocabulary_.get(token)
                if idx is None:
                    continue
                if dense:
                    X[i, idx] = float(value)
                else:
                    rows.append(i)
                    cols.append(idx)
                    data.append(float(value))
        if dense:
            return X
        return SparseCOO(
            row=np.asarray(rows, dtype=int),
            col=np.asarray(cols, dtype=int),
            data=np.asarray(data, dtype=float),
            shape=(n_rows, n_cols),
        )

    def fit_transform(self, texts, y=None, dense=True):
        return self.fit(texts, y).transform(texts, dense=dense)


class TfidfVectorizer(CountVectorizer):
    """CountVectorizer with inverse-document-frequency reweighting."""

    def __init__(self, tokenizer=None, ngram_range=(1, 1), lowercase=True, max_features=None, smooth_idf=True, norm="l2", stop_words=None):
        super().__init__(tokenizer=tokenizer, ngram_range=ngram_range, lowercase=lowercase, max_features=max_features, stop_words=stop_words)
        self.smooth_idf = bool(smooth_idf)
        self.norm = norm
        self.idf_ = None

    def fit(self, texts, y=None):
        texts = list(texts)
        self.stop_words_ = self._prepare_stop_words()
        counts = Counter()
        doc_counts = Counter()
        for text in texts:
            feats = self._feats(text, stop_words=self.stop_words_)
            counts.update(feats)
            doc_counts.update(set(feats))

        self._build_vocab(counts)

        doc_freq = np.zeros(len(self.vocabulary_), dtype=float)
        for token, idx in self.vocabulary_.items():
            doc_freq[idx] = float(doc_counts.get(token, 0))

        n_docs = max(len(texts), 1)
        if self.smooth_idf:
            self.idf_ = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
        else:
            self.idf_ = np.log(np.maximum(n_docs / np.maximum(doc_freq, 1.0), 1e-12)) + 1.0
        return self

    def transform(self, texts, dense=True):
        counts = super().transform(texts, dense=dense)
        if self.idf_ is None:
            raise RuntimeError("Vectorizer has not been fit yet")
        if dense:
            tfidf = counts * self.idf_[None, :]
            if self.norm is None:
                return tfidf
            if self.norm not in {"l1", "l2"}:
                raise ValueError("norm must be one of: None, l1, l2")
            if self.norm == "l1":
                scale = np.sum(np.abs(tfidf), axis=1, keepdims=True)
            else:
                scale = np.sqrt(np.sum(tfidf ** 2, axis=1, keepdims=True))
            scale = np.where(scale == 0.0, 1.0, scale)
            return tfidf / scale

        if self.norm not in {None, "l1", "l2"}:
            raise ValueError("norm must be one of: None, l1, l2")
        data = counts.data * self.idf_[counts.col]
        if self.norm is not None:
            if self.norm == "l1":
                scale = np.bincount(counts.row, weights=np.abs(data), minlength=counts.shape[0])
            else:
                scale = np.bincount(counts.row, weights=data ** 2, minlength=counts.shape[0])
                scale = np.sqrt(scale)
            scale = np.where(scale == 0.0, 1.0, scale)
            data = data / scale[counts.row]
        return SparseCOO(row=counts.row, col=counts.col, data=data, shape=counts.shape)


class HashingVectorizer:
    """Fixed-width hashed n-gram vectorizer."""

    def __init__(self, tokenizer=None, ngram_range=(1, 1), lowercase=True, n_features=2**18, norm="l2", alternate_sign=False):
        self.tokenizer = tokenizer or SimpleTokenizer(lowercase=False)
        self.ngram_range = tuple(ngram_range)
        self.lowercase = bool(lowercase)
        self.n_features = int(n_features)
        self.norm = norm
        self.alternate_sign = bool(alternate_sign)
        self.n_features_ = self.n_features

    def _tokens(self, text):
        if self.lowercase and isinstance(text, str):
            text = text.lower()
        return self.tokenizer(text)

    def _feats(self, text):
        tokens = self._tokens(text)
        feats = []
        lo, hi = self.ngram_range
        for n in range(int(lo), int(hi) + 1):
            if n == 1:
                feats.extend(tokens)
            else:
                feats.extend([" ".join(ng) for ng in word_ngrams(tokens, n=n)])
        return feats

    def _hash(self, token):
        data = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(data, "little", signed=False)

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        texts = list(texts)
        X = np.zeros((len(texts), self.n_features), dtype=float)
        for i, text in enumerate(texts):
            counts = Counter(self._feats(text))
            for token, value in counts.items():
                h = self._hash(token)
                idx = h % self.n_features
                if self.alternate_sign:
                    val = -1.0 if (h >> 1) & 1 else 1.0
                else:
                    val = 1.0
                X[i, idx] += val * float(value)

        if self.norm is None:
            return X
        if self.norm not in {"l1", "l2"}:
            raise ValueError("norm must be one of: None, l1, l2")
        if self.norm == "l1":
            scale = np.sum(np.abs(X), axis=1, keepdims=True)
        else:
            scale = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        scale = np.where(scale == 0.0, 1.0, scale)
        return X / scale

    def fit_transform(self, texts, y=None):
        return self.fit(texts, y).transform(texts)
