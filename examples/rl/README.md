# Reinforcement Learning Tutorial

[`q_learning_demo.py`](q_learning_demo.py) trains tabular Q-learning in a
five-by-five `GridWorld`, evaluates without exploration, and prints the learned
greedy route.

## Run

```bash
python -m pip install "mastermlx==0.1.15"
python examples/rl/q_learning_demo.py
```

## Environment and agent interfaces

```python
from mastermlx.rl import GridWorld, QLearningAgent, evaluate, train_tabular

env = GridWorld(rows=5, cols=5, goal=(4, 4), random_state=0)
agent = QLearningAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    random_state=0,
)
rewards = train_tabular(env, agent, episodes=1000)
mean_reward = evaluate(env, agent, episodes=100)
```

`GridWorld.reset()` returns a state. `step(action)` returns
`(next_state, reward, done)`. Tabular agents implement `select_action` and
`update`, while the runner owns the episode loop. `evaluate` temporarily uses
a greedy policy and restores the exploration rate afterward.

Actions are `0=up`, `1=right`, `2=down`, and `3=left`; states are flattened as
`row * n_cols + col`. The package also exports SARSA, double Q-learning, DQN,
and REINFORCE. See the
[`RL API index`](../API_REFERENCE.md#bandits-and-reinforcement-learning).
