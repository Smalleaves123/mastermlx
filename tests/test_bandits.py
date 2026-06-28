import numpy as np

from mastermlx.bandits import (
    BernoulliThompsonSampling,
    EpsilonGreedyBandit,
    Exp3Bandit,
    LinUCBBandit,
    LinearThompsonBandit,
    SoftmaxBandit,
    UCBBandit,
)


def test_epsilon_greedy_bandit_updates_estimates():
    bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.0, initial_value=0.0, random_state=0)
    bandit.update(1, 1.0)
    bandit.update(1, 0.0)
    assert bandit.counts_[1] == 2
    assert np.isclose(bandit.q_values_[1], 0.5)


def test_ucb_bandit_prefers_unseen_arms_first():
    bandit = UCBBandit(n_arms=3)
    assert bandit.select_arm() == 0
    bandit.update(0, 1.0)
    assert bandit.select_arm() == 1


def test_thompson_sampling_updates_beta_posteriors():
    bandit = BernoulliThompsonSampling(n_arms=2, random_state=0)
    bandit.update(0, 1.0)
    bandit.update(1, 0.0)
    assert bandit.alpha_[0] > 1.0
    assert bandit.beta_[1] > 1.0


def test_softmax_bandit_prefers_high_value_arm():
    bandit = SoftmaxBandit(n_arms=3, temperature=0.2, random_state=0)
    bandit.q_values_ = np.array([0.0, 1.0, 3.0], dtype=float)
    probs = bandit.arm_probabilities()
    assert np.isclose(probs.sum(), 1.0)
    assert probs[2] > probs[1] > probs[0]


def test_linucb_updates_contextual_estimates():
    bandit = LinUCBBandit(n_arms=2, n_features=2, alpha=0.1)
    context_a = np.array([1.0, 0.0])
    context_b = np.array([0.0, 1.0])

    for _ in range(25):
        bandit.update(0, context_a, 1.0)
        bandit.update(1, context_b, 1.0)

    pred_a = bandit.predict_reward(context_a)
    pred_b = bandit.predict_reward(context_b)
    assert pred_a[0] > pred_a[1]
    assert pred_b[1] > pred_b[0]


def test_linucb_select_arm_returns_scores():
    bandit = LinUCBBandit(n_arms=2, n_features=2, alpha=0.5)
    bandit.update(1, np.array([1.0, 1.0]), 2.0)
    arm, scores = bandit.select_arm(np.array([1.0, 1.0]), return_scores=True)
    assert arm in {0, 1}
    assert scores.shape == (2,)


def test_exp3_probabilities_sum_to_one_and_update_weights():
    bandit = Exp3Bandit(n_arms=3, gamma=0.2, random_state=0)
    arm, probs = bandit.select_arm(return_probs=True)
    old_weight = bandit.weights_[arm]
    bandit.update(arm, 1.0, probs=probs)
    assert np.isclose(np.sum(probs), 1.0)
    assert bandit.weights_[arm] > old_weight


def test_linear_thompson_learns_context_preferences():
    bandit = LinearThompsonBandit(n_arms=2, n_features=2, v=0.1, random_state=0)
    context_a = np.array([1.0, 0.0])
    context_b = np.array([0.0, 1.0])

    for _ in range(30):
        bandit.update(0, context_a, 1.0)
        bandit.update(1, context_b, 1.0)

    pred_a = bandit.predict_reward(context_a)
    pred_b = bandit.predict_reward(context_b)
    assert pred_a[0] > pred_a[1]
    assert pred_b[1] > pred_b[0]


def test_linear_thompson_select_arm_returns_samples():
    bandit = LinearThompsonBandit(n_arms=2, n_features=2, v=0.2, random_state=0)
    bandit.update(0, np.array([1.0, 0.0]), 1.0)
    arm, samples = bandit.select_arm(np.array([1.0, 0.0]), return_samples=True)
    assert arm in {0, 1}
    assert samples.shape == (2,)
