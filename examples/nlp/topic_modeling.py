"""Discover topics with CountVectorizer and variational LDA.

Run from the repository root:

    python examples/nlp/topic_modeling.py
"""

import numpy as np

from mastermlx.nlp import CountVectorizer, NLP_LDA


DOCUMENTS = [
    "robot arm motion joint controller trajectory",
    "robot gripper joint motion planning",
    "controller plans robot arm trajectory",
    "language model token text vocabulary",
    "text document topic word model",
    "token vocabulary for language document",
]


def top_words(model, feature_names, n_words=5):
    """Return the highest-probability terms for each fitted topic."""

    names = np.asarray(feature_names)
    topics = []
    for weights in model.components_:
        indices = np.argsort(weights)[-n_words:][::-1]
        topics.append(names[indices].tolist())
    return topics


def main():
    vectorizer = CountVectorizer(stop_words="english")
    counts = vectorizer.fit_transform(DOCUMENTS)

    model = NLP_LDA(
        n_topics=2,
        alpha=0.05,
        eta=0.01,
        max_iter=50,
        tol=1e-5,
        random_state=7,
    )
    document_topics = model.fit_transform(counts)

    for index, words in enumerate(
        top_words(model, vectorizer.feature_names_),
        start=1,
    ):
        print(f"topic {index}: {', '.join(words)}")
    print("document-topic shape:", document_topics.shape)
    print("row sums:", np.round(document_topics.sum(axis=1), 6))
    print("iterations:", model.n_iter_)
    print("perplexity:", f"{model.perplexity(counts):.3f}")


if __name__ == "__main__":
    main()
