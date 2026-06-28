import numpy as np

from mastermlx.nlp import CountVectorizer
from mastermlx.probabilistic import BernoulliNB, GaussianNB, MultinomialNB


def test_gaussian_nb_fits_simple_data():
    X = np.array([
        [1.0, 1.0],
        [1.2, 0.8],
        [3.0, 3.0],
        [3.2, 2.8],
    ])
    y = np.array([0, 0, 1, 1])

    model = GaussianNB()
    model.fit(X, y)
    pred = model.predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0


def test_gaussian_nb_predict_single_sample():
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [2.0, 2.0],
        [2.1, 1.9],
    ])
    y = np.array([0, 0, 1, 1])

    model = GaussianNB().fit(X, y)
    pred = model.predict([2.0, 2.0])

    assert pred == 1


def test_bernoulli_nb_handles_binary_text_features():
    docs = ["cat sat", "cat purrs", "dog barked", "dog ran"]
    y = np.array([0, 0, 1, 1])

    X = CountVectorizer().fit_transform(docs)
    model = BernoulliNB(alpha=1.0, binarize=0.0).fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X[:1])

    assert np.array_equal(pred, y)
    assert proba.shape == (2,)
    assert model.score(X, y) == 1.0


def test_multinomial_nb_handles_count_text_features():
    docs = ["cat cat sat", "cat purrs", "dog barked", "dog dog ran"]
    y = np.array([0, 0, 1, 1])

    X = CountVectorizer().fit_transform(docs)
    model = MultinomialNB(alpha=1.0).fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X[:1])

    assert np.array_equal(pred, y)
    assert proba.shape == (2,)
    assert model.score(X, y) == 1.0
