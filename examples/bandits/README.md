# Multi-Armed Bandit Tutorial

[`bandit_comparison.py`](bandit_comparison.py) compares epsilon-greedy, upper
confidence bound, and Bernoulli Thompson sampling on the same four arms.

## Run

```bash
python -m pip install "mastermlx==0.1.15"
python examples/bandits/bandit_comparison.py
```

## Interface

```python
from mastermlx.bandits import EpsilonGreedyBandit

policy = EpsilonGreedyBandit(n_arms=4, epsilon=0.1, random_state=0)
for observation in stream:
    arm = policy.select_arm()
    reward = observe_reward(arm, observation)
    policy.update(arm, reward)
```

The application owns the interaction loop and reward generation. The policy
only chooses an arm and updates its statistics. Learned state is exposed with
trailing underscores such as `q_values_`, `counts_`, and `total_reward_`.

Use `BernoulliThompsonSampling` only for binary rewards. `LinUCBBandit` and
`LinearThompsonBandit` accept contextual features; `Exp3Bandit` targets
adversarial rewards. See the
[`bandit API index`](../API_REFERENCE.md#bandits-and-reinforcement-learning).
