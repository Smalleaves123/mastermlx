# NLP Tutorials

This directory covers two distinct workflows:

- [`text_classify.py`](text_classify.py): TF-IDF features plus logistic
  regression and visual diagnostics.
- [`topic_modeling.py`](topic_modeling.py): integer count features plus latent
  Dirichlet allocation.

## Run

```bash
python examples/nlp/topic_modeling.py
python -m pip install "mastermlx[viz]==0.1.15"
python examples/nlp/text_classify.py
```

## Text feature interface

```python
from mastermlx.nlp import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(train_documents)
X_test = vectorizer.transform(test_documents)
feature_names = vectorizer.feature_names_
```

Fit the vectorizer only on training text to avoid feature and IDF leakage.

## Topic-model interface and the LDA name

```python
from mastermlx.nlp import CountVectorizer, NLP_LDA

counts = CountVectorizer().fit_transform(documents)
topics = NLP_LDA(n_topics=3, random_state=0).fit_transform(counts)
```

Topic LDA requires a non-negative integer document-term matrix. In 0.1.15,
`mastermlx.LDA` is intentionally unavailable because it is ambiguous. Use
`mastermlx.nlp.NLP_LDA` for topic modeling and
`mastermlx.probabilistic.DiscriminantLDA` for classification.

See the [`NLP API index`](../API_REFERENCE.md#nlp) for tokenizers, sequence
helpers, hashing, and n-gram language models.
