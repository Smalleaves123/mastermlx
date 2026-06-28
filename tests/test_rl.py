import numpy as np

from mastermlx.rl import QLearningAgent


def test_q_learning_agent_updates_q_values():
    agent = QLearningAgent(n_states=3, n_actions=2, alpha=0.5, gamma=0.9, epsilon=0.0, random_state=0)
    agent.update(state=0, action=1, reward=1.0, next_state=2, done=False)
    assert agent.q_table_[0, 1] > 0.0
    action = agent.select_action(0)
    assert action in {0, 1}
    values = agent.value_function()
    assert values.shape == (3,)
