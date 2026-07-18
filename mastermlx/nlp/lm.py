from __future__ import annotations

import math
from collections import Counter, defaultdict
import numpy as np

from .tok import SimpleTokenizer, word_ngrams


class NGramLanguageModel:
    """Smoothed n-gram language model with simple text generation."""

    def __init__(self, n=2, smoothing=1.0, tokenizer=None):
        self.n = int(n)
        self.smoothing = float(smoothing)
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.context_counts_ = defaultdict(Counter)
        self.context_totals_: Counter = Counter()
        self.vocab_ = set()
        self.vocab_list_ = []
        self.vocab_size_ = 0
        self.fitted_ = False

    def fit(self, texts):
        texts = list(texts)
        if self.n < 1:
            raise ValueError("n must be at least 1")
        self.context_counts_.clear()
        self.context_totals_.clear()
        self.vocab_.clear()

        for text in texts:
            tokens = self.tokenizer(text)
            self.vocab_.update(tokens)
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for gram in word_ngrams(padded, n=self.n):
                context, token = gram[:-1], gram[-1]
                self.context_counts_[context][token] += 1
                self.context_totals_[context] += 1
                self.vocab_.add(token)

        self.vocab_.update({"<s>", "</s>"})
        self.vocab_list_ = sorted(self.vocab_)
        self.vocab_size_ = len(self.vocab_list_)
        self.fitted_ = True
        return self

    def _context(self, tokens):
        tokens = list(tokens)
        if len(tokens) >= self.n - 1:
            return tuple(tokens[-(self.n - 1) :])
        return tuple(["<s>"] * (self.n - 1 - len(tokens)) + tokens)

    def probability(self, token, context):
        if not self.fitted_:
            raise RuntimeError("Model has not been fit yet")
        context = tuple(context)
        counts = self.context_counts_.get(context, Counter())
        total = self.context_totals_.get(context, 0)
        return (counts.get(token, 0) + self.smoothing) / (total + self.smoothing * self.vocab_size_)

    def sequence_log_prob(self, text):
        if not self.fitted_:
            raise RuntimeError("Model has not been fit yet")
        tokens = self.tokenizer(text)
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        logp = 0.0
        for gram in word_ngrams(padded, n=self.n):
            context, token = gram[:-1], gram[-1]
            logp += math.log(self.probability(token, context))
        return logp

    def generate(self, max_tokens=20, seed=None):
        if not self.fitted_:
            raise RuntimeError("Model has not been fit yet")
        rng = np.random.default_rng(seed)
        context = tuple(["<s>"] * (self.n - 1))
        out = []
        for _ in range(int(max_tokens)):
            counts = self.context_counts_.get(context, Counter())
            if not counts:
                token = rng.choice(self.vocab_list_)
            else:
                seen_tokens = list(counts.keys())
                seen_weights = np.asarray([count + self.smoothing for count in counts.values()], dtype=float)
                seen_mass = float(seen_weights.sum()) / float(self.context_totals_.get(context, 0) + self.smoothing * self.vocab_size_)
                if rng.random() < seen_mass:
                    token = rng.choice(seen_tokens, p=seen_weights / seen_weights.sum())
                else:
                    seen = set(seen_tokens)
                    if len(seen) * 2 < self.vocab_size_:
                        while True:
                            token = rng.choice(self.vocab_list_)
                            if token not in seen:
                                break
                    else:
                        unseen = [tok for tok in self.vocab_list_ if tok not in seen]
                        token = rng.choice(unseen)
            if token == "</s>":
                break
            out.append(token)
            context = self._context((*context, token))
        return out


NGramLM = NGramLanguageModel
