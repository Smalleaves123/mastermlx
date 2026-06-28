from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def normalize_text(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.lower()
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def word_ngrams(tokens, n=2, pad_left=False, pad_right=False, pad_token="<s>"):
    tokens = list(tokens)
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    if pad_left:
        tokens = [pad_token] * (n - 1) + tokens
    if pad_right:
        tokens = tokens + [pad_token] * (n - 1)
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class SimpleTokenizer:
    """Tokenize text into lowercase word-like tokens."""

    def __init__(self, lowercase=True, strip_punctuation=True, min_length=1):
        self.lowercase = bool(lowercase)
        self.strip_punctuation = bool(strip_punctuation)
        self.min_length = int(min_length)

    def tokenize(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self.lowercase:
            text = text.lower()
        if self.strip_punctuation:
            tokens = _WORD_RE.findall(text)
        else:
            tokens = text.split()
        return [tok for tok in tokens if len(tok) >= self.min_length]

    def __call__(self, text):
        return self.tokenize(text)


class CharTokenizer:
    """Tokenize text into individual characters."""

    def __init__(self, lowercase=True, strip_spaces=True, min_length=1):
        self.lowercase = bool(lowercase)
        self.strip_spaces = bool(strip_spaces)
        self.min_length = int(min_length)

    def tokenize(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self.lowercase:
            text = text.lower()
        chars = list(text)
        if self.strip_spaces:
            chars = [ch for ch in chars if not ch.isspace()]
        return [ch for ch in chars if len(ch) >= self.min_length]

    def __call__(self, text):
        return self.tokenize(text)
