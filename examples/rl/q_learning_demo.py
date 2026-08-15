"""Train and evaluate tabular Q-learning in GridWorld.

Run from the repository root:

    python examples/rl/q_learning_demo.py
"""

import numpy as np

from mastermlx.rl import GridWorld, QLearningAgent, evaluate, train_tabular


ACTION_NAMES = ("up", "right", "down", "left")


def greedy_path(env, agent, max_steps=30):
    """Roll out the learned greedy policy and return state/action traces."""

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        state = env.reset()
        states = [state]
        actions = []
        for _ in range(max_steps):
            action = agent.select_action(state)
            state, _, done = env.step(action)
            actions.append(ACTION_NAMES[action])
            states.append(state)
            if done:
                break
    finally:
        agent.epsilon = old_epsilon
    return states, actions


def main():
    env = GridWorld(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        walls={(1, 1), (1, 2), (2, 2), (3, 2)},
        random_state=0,
    )
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.25,
        gamma=0.95,
        epsilon=0.15,
        random_state=0,
    )

    rewards = train_tabular(env, agent, episodes=1200, max_steps=100)
    mean_reward = evaluate(env, agent, episodes=100, max_steps=100)
    states, actions = greedy_path(env, agent)

    print(f"last-100 training reward: {np.mean(rewards[-100:]):.3f}")
    print(f"greedy evaluation reward: {mean_reward:.3f}")
    print("state path:", states)
    print("actions:", " -> ".join(actions))
    print("q_table_ shape:", agent.q_table_.shape)


if __name__ == "__main__":
    main()
