import numpy as np

from mastermlx.nlp import (
    CharTokenizer,
    CountVectorizer,
    HashingVectorizer,
    NGramLanguageModel,
    SparseCOO,
    SeqPad,
    SimpleTokenizer,
    TextSeq,
    Vocab,
    TfidfVectorizer,
    normalize_text,
    split_sentences,
    word_ngrams,
)


def test_nlp_normalization_and_tokenization():
    assert normalize_text("  Hello   WORLD  ") == "hello world"
    assert split_sentences("Hi there. How are you? Fine!") == ["Hi there.", "How are you?", "Fine!"]

    tok = SimpleTokenizer()
    assert tok("Hello, world!") == ["hello", "world"]

    ctok = CharTokenizer()
    assert ctok("Ab c") == ["a", "b", "c"]


def test_nlp_ngrams_and_vectorizer():
    tokens = ["a", "b", "c"]
    assert word_ngrams(tokens, n=2) == [("a", "b"), ("b", "c")]

    docs = ["The cat sat", "The dog sat"]
    vec = CountVectorizer(ngram_range=(1, 2), stop_words={"the"})
    X = vec.fit_transform(docs)
    sparse = vec.fit_transform(docs, dense=False)

    assert X.shape[0] == 2
    assert "the" not in vec.vocabulary_
    assert "the cat" not in vec.vocabulary_
    assert np.array_equal(X, np.asarray(sparse))


def test_nlp_sparse_coo_is_public_and_round_trips():
    X = SparseCOO(
        row=np.array([0, 1, 1]),
        col=np.array([0, 0, 2]),
        data=np.array([1.0, 2.0, 3.0]),
        shape=(2, 3),
    )

    dense = X.toarray()

    assert dense.shape == (2, 3)
    assert np.array_equal(dense, np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 3.0]]))


def test_nlp_tfidf_vectorizer_builds_weighted_features():
    docs = ["the cat sat", "the dog sat", "the cat ate fish"]
    vec = TfidfVectorizer(ngram_range=(1, 1))
    X = vec.fit_transform(docs)

    assert X.shape[0] == 3
    assert X.shape[1] == len(vec.vocabulary_)
    assert vec.idf_.shape[0] == X.shape[1]
    assert np.allclose(np.sqrt(np.sum(X ** 2, axis=1)), 1.0, atol=1e-8)


def test_nlp_hashing_vectorizer_works_without_fit():
    docs = ["the cat sat", "the dog sat"]

    vec = HashingVectorizer(n_features=32, norm="l2")
    X1 = vec.transform(docs)
    X2 = vec.fit_transform(docs)

    assert X1.shape == (2, 32)
    assert np.allclose(X1, X2)
    assert np.allclose(np.sqrt(np.sum(X1 ** 2, axis=1)), 1.0, atol=1e-8)


def test_nlp_ngram_language_model_scores_and_generates():
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat ate fish",
    ]

    model = NGramLanguageModel(n=2, smoothing=1.0)
    model.fit(corpus)

    lp = model.sequence_log_prob("the cat sat on the mat")
    assert np.isfinite(lp)
    assert model.vocab_size_ == len(model.vocab_)

    generated = model.generate(max_tokens=5, seed=0)
    assert isinstance(generated, list)


def test_nlp_vocab_and_text_seq_build_sequence_features():
    docs = ["the cat sat", "the dog sat"]

    vocab = Vocab(min_freq=1).fit(docs)
    ids = vocab.encode("the cat")
    back = vocab.decode(ids)

    seq = TextSeq(max_len=4).fit_transform(docs)

    assert "the" in vocab.stoi_
    assert ids.ndim == 1
    assert back[:2] == ["the", "cat"]
    assert seq.shape == (2, 4)


def test_nlp_seq_pad_truncates_cleanly():
    pad = SeqPad(max_len=3, pad_id=0, trunc="left")

    out = pad([1, 2, 3, 4, 5])

    assert np.array_equal(out, np.array([3, 4, 5]))
