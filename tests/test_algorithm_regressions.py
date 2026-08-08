import copy

import numpy as np
import pytest

from mastermlx.bandits import Exp3Bandit
from mastermlx.clustering import KMeans, MiniBatchKMeans
from mastermlx.ensemble.hist_gb import _bin_data
from mastermlx.nlp import CountVectorizer, LDA
from mastermlx.rl import DQNAgent, REINFORCEAgent, evaluate
from mastermlx.semi_supervised import LabelSpreading
from mastermlx.semi_supervised._core import rbf_affinity, row_norm, sym_norm
from mastermlx.vision import histogram_of_oriented_gradients


def _topic_corpus(random_state=0):
    rng = np.random.default_rng(random_state)
    documents = []
    groups = []
    for topic in range(3):
        for _ in range(8):
            counts = rng.poisson(0.1, size=15)
            counts[topic * 5 : (topic + 1) * 5] += rng.poisson(8.0, size=5)
            documents.append(counts)
            groups.append(topic)
    return np.asarray(documents, dtype=float), np.asarray(groups)


def test_lda_runs_requested_iterations_and_replaces_m_step_statistics():
    X, _ = _topic_corpus()
    short = LDA(n_topics=3, max_iter=4, tol=0.0, random_state=7).fit(X)
    long = LDA(n_topics=3, max_iter=12, tol=0.0, random_state=7).fit(X)

    assert short.n_iter_ == 4
    assert long.n_iter_ == 12
    assert not np.allclose(short.components_, long.components_)
    assert np.allclose(long.components_.sum(axis=1), 1.0)
    assert np.allclose(long.doc_topic_.sum(axis=1), 1.0)
    assert np.isfinite(long.bound_)


def test_lda_recovers_separated_synthetic_topics():
    X, groups = _topic_corpus()
    model = LDA(n_topics=3, max_iter=40, tol=1e-7, random_state=0).fit(X)
    group_topics = [
        int(np.argmax(np.mean(model.doc_topic_[groups == group], axis=0)))
        for group in range(3)
    ]

    assert len(set(group_topics)) == 3
    for group, topic in enumerate(group_topics):
        top_words = set(np.argsort(model.components_[topic])[-5:])
        expected_words = set(range(group * 5, (group + 1) * 5))
        assert len(top_words & expected_words) >= 4


class _OneStepEnvironment:
    def reset(self):
        return np.array([1.0])

    def step(self, action):
        return np.array([1.0]), 1.0, True


def test_dqn_evaluate_is_side_effect_free():
    agent = DQNAgent(
        1,
        2,
        hidden_sizes=(2,),
        batch_size=1,
        epsilon=0.5,
        random_state=0,
    )
    weights = copy.deepcopy(agent.weights_)
    target_weights = copy.deepcopy(agent.target_weights_)
    epsilon = agent.epsilon

    evaluate(_OneStepEnvironment(), agent, episodes=3, max_steps=1)

    assert len(agent.memory) == 0
    assert agent._step_count == 0
    assert agent.epsilon == epsilon
    for before, after in zip(weights, agent.weights_):
        assert all(np.array_equal(left, right) for left, right in zip(before, after))
    for before, after in zip(target_weights, agent.target_weights_):
        assert all(np.array_equal(left, right) for left, right in zip(before, after))


def test_dqn_hidden_gradient_uses_forward_pass_weights():
    agent = DQNAgent(
        2,
        1,
        hidden_sizes=(2,),
        lr=0.5,
        batch_size=1,
        target_update=100,
        random_state=0,
    )
    W1 = np.array([[0.5, -0.25], [0.2, 0.4]])
    b1 = np.array([0.1, 0.2])
    W2 = np.array([[0.3], [-0.6]])
    b2 = np.array([0.05])
    agent.weights_ = [(W1.copy(), b1.copy()), (W2.copy(), b2.copy())]
    agent.target_weights_ = copy.deepcopy(agent.weights_)
    state = np.array([1.2, 0.7])
    reward = 0.8
    hidden = np.maximum(0.0, state @ W1 + b1)
    prediction = hidden @ W2 + b2
    output_grad = prediction - reward
    hidden_grad = (output_grad @ W2.T) * (hidden > 0.0)
    expected_W1 = W1 - agent.lr * np.outer(state, hidden_grad)

    agent.store(state, 0, reward, state, True)
    agent.update()

    assert np.allclose(agent.weights_[0][0], expected_W1)


def _correct_reinforce_update(layers, episode, lr, gamma):
    layers = copy.deepcopy(layers)
    returns = np.zeros(len(episode))
    value = 0.0
    for index in range(len(episode) - 1, -1, -1):
        value = episode[index][2] + gamma * value
        returns[index] = value
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    for sample, (state, action, _) in enumerate(episode):
        activations = [np.asarray(state, dtype=float)]
        hidden = activations[0]
        for index, (W, b) in enumerate(layers):
            hidden = hidden @ W + b
            if index < len(layers) - 1:
                hidden = np.maximum(0.0, hidden)
            else:
                exponent = np.exp(hidden - np.max(hidden))
                hidden = exponent / np.sum(exponent)
            activations.append(hidden)
        gradient = activations[-1].copy()
        gradient[action] -= 1.0
        gradient *= returns[sample]
        for index in range(len(layers) - 1, -1, -1):
            W, b = layers[index]
            previous = activations[index]
            next_gradient = (
                (gradient @ W.T) * (previous > 0.0) if index > 0 else None
            )
            layers[index] = (
                W - lr * np.outer(previous, gradient),
                b - lr * gradient,
            )
            if index > 0:
                gradient = next_gradient
    return layers


def test_reinforce_hidden_gradient_uses_forward_pass_weights():
    agent = REINFORCEAgent(2, 2, hidden_sizes=(2,), lr=0.5, gamma=0.9, random_state=0)
    agent.layers = [
        (np.array([[0.5, 0.1], [0.2, 0.4]]), np.array([0.2, 0.1])),
        (np.array([[0.3, -0.2], [-0.6, 0.5]]), np.array([0.05, -0.05])),
    ]
    episode = [
        (np.array([1.0, 0.5]), 0, 1.0),
        (np.array([0.4, 1.2]), 1, 0.0),
    ]
    expected = _correct_reinforce_update(agent.layers, episode, agent.lr, agent.gamma)

    agent.update_episode(episode)

    for expected_layer, actual_layer in zip(expected, agent.layers):
        assert all(
            np.allclose(expected_value, actual_value)
            for expected_value, actual_value in zip(expected_layer, actual_layer)
        )


def test_histogram_validation_binning_reuses_training_edges():
    training = np.array([[0.0], [1.0], [2.0], [100.0]])
    training_bins, edges = _bin_data(training, n_bins=3)
    repeated_bins, _ = _bin_data(training, edges=edges)
    validation_bins, _ = _bin_data(np.array([[0.5], [50.0]]), edges=edges)

    assert np.array_equal(training_bins, repeated_bins)
    assert np.all(validation_bins >= 0)
    assert np.all(validation_bins <= np.max(training_bins))


def test_label_spreading_matches_standard_soft_clamped_fixed_point():
    X = np.array([[0.0], [0.8], [2.5]])
    y = np.array([0, -1, 1])
    alpha = 0.3
    model = LabelSpreading(
        gamma=1.2,
        alpha=alpha,
        max_iter=2000,
        tol=1e-12,
    ).fit(X, y)

    Y = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    S = sym_norm(rbf_affinity(X, 1.2))
    expected = Y.copy()
    for _ in range(5000):
        updated = alpha * (S @ expected) + (1.0 - alpha) * Y
        if np.max(np.abs(updated - expected)) < 1e-14:
            expected = updated
            break
        expected = updated
    expected = row_norm(expected)

    assert np.allclose(model.label_distributions_, expected, atol=1e-9)


@pytest.mark.parametrize("estimator", [KMeans(n_clusters=2), MiniBatchKMeans(n_clusters=2)])
def test_kmeans_variants_accept_generator_random_state(estimator):
    estimator.random_state = np.random.default_rng(4)
    estimator.n_init = 2
    estimator.max_iter = 5
    X = np.array([[0.0], [0.1], [3.0], [3.1]])

    labels = estimator.fit_predict(X)

    assert labels.shape == (4,)


def test_english_stop_words_are_not_treated_as_a_literal_token():
    vectorizer = CountVectorizer(stop_words="english").fit(["the english cat"])

    assert "the" not in vectorizer.vocabulary_
    assert "english" in vectorizer.vocabulary_
    with pytest.raises(ValueError, match="must be 'english'"):
        CountVectorizer(stop_words="unsupported").fit(["text"])


def test_hog_constant_image_is_a_finite_zero_descriptor():
    descriptor = histogram_of_oriented_gradients(np.zeros((16, 16)))

    assert np.all(np.isfinite(descriptor))
    assert np.array_equal(descriptor, np.zeros_like(descriptor))


def test_exp3_rejects_rewards_outside_its_mathematical_domain():
    bandit = Exp3Bandit(n_arms=2, gamma=0.2, random_state=0)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bandit.update(0, 1.1)
    with pytest.raises(ValueError, match="finite"):
        bandit.update(0, np.nan)
    assert np.all(np.isfinite(bandit.weights_))
