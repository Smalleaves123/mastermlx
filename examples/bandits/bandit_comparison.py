"""Compare three policies on a stationary Bernoulli bandit.

Run from the repository root:

    python examples/bandits/bandit_comparison.py
"""

import numpy as np

from mastermlx.bandits import (
    BernoulliThompsonSampling,
    EpsilonGreedyBandit,
    UCBBandit,
)


ARM_PROBABILITIES = np.array([0.10, 0.25, 0.55, 0.80])


def run_policy(policy, random_state, steps=2000):
    """Return cumulative reward and optimal-arm selection rate."""

    rng = np.random.default_rng(random_state)
    selected = np.zeros(policy.n_arms, dtype=int)
    rewards = 0.0

    for _ in range(steps):
        arm = policy.select_arm()
        reward = float(rng.random() < ARM_PROBABILITIES[arm])
        policy.update(arm, reward)
        selected[arm] += 1
        rewards += reward

    optimal_arm = int(np.argmax(ARM_PROBABILITIES))
    return rewards, selected[optimal_arm] / steps


def estimated_values(policy):
    """Return empirical means or Bernoulli posterior means."""

    if hasattr(policy, "q_values_"):
        return policy.q_values_
    return policy.alpha_ / (policy.alpha_ + policy.beta_)


def main():
    policies = {
        "epsilon-greedy": EpsilonGreedyBandit(
            n_arms=4,
            epsilon=0.1,
            random_state=1,
        ),
        "UCB": UCBBandit(n_arms=4, c=1.5),
        "Thompson": BernoulliThompsonSampling(n_arms=4, random_state=2),
    }

    print("true arm probabilities:", ARM_PROBABILITIES)
    for index, (name, policy) in enumerate(policies.items()):
        reward, optimal_rate = run_policy(policy, random_state=100 + index)
        print(
            f"{name:16s} reward={reward:6.0f} "
            f"optimal_arm_rate={optimal_rate:.1%} "
            f"estimates={np.round(estimated_values(policy), 3)}"
        )


if __name__ == "__main__":
    main()
