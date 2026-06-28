import numpy as np

from mastermlx.rl import (
    DoubleQLearningAgent, DQNAgent, GridWorld, QLearningAgent,
    REINFORCEAgent, SARSAAgent, evaluate, run_episode, train_tabular,
)


# --- GridWorld ---
def test_gridworld_reaches_goal():
    env = GridWorld(rows=4, cols=4, start=(0, 0), goal=(3, 3), random_state=0)
    s = env.reset()
    assert s == 0
    for _ in range(20):
        s, r, done = env.step(2)  # move down
        s, r, done = env.step(1)  # move right
        if done:
            break
    assert done


# --- Q-Learning ---
def test_qlearning_solves_small_grid():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), random_state=0)
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.5, gamma=0.9,
                           epsilon=0.3, random_state=0)
    for _ in range(500):
        s = env.reset()
        for _ in range(50):
            a = agent.select_action(s)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns, done)
            if done:
                break
            s = ns
    # After training, greedy policy should reach goal
    s = env.reset()
    steps = 0
    for _ in range(20):
        a = agent.select_action(s)
        a = int(np.argmax(agent.q_table_[s]))  # greedy
        s, r, done = env.step(a)
        steps += 1
        if done:
            break
    assert done and steps <= 10


# --- SARSA ---
def test_sarsa_solves_small_grid():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), random_state=0)
    agent = SARSAAgent(env.n_states, env.n_actions, alpha=0.5, gamma=0.9,
                       epsilon=0.3, random_state=0)
    for _ in range(500):
        s = env.reset()
        a = agent.select_action(s)
        for _ in range(50):
            ns, r, done = env.step(a)
            na = agent.select_action(ns)
            agent.update(s, a, r, ns, na, done)
            if done:
                break
            s, a = ns, na
    s = env.reset()
    for _ in range(20):
        a = int(np.argmax(agent.q_table_[s]))
        s, r, done = env.step(a)
        if done:
            break
    assert done


# --- REINFORCE ---
def test_reinforce_runs_on_cartpole_style():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), random_state=0)
    agent = REINFORCEAgent(n_features=env.n_states, n_actions=env.n_actions,
                           hidden_sizes=(16,), lr=0.01, gamma=0.99, random_state=0)
    # One-hot encode state
    def encode(s):
        v = np.zeros(env.n_states)
        v[s] = 1.0
        return v

    for _ in range(200):
        episode = []
        s = env.reset()
        for _ in range(50):
            a = agent.select_action(encode(s))
            ns, r, done = env.step(a)
            episode.append((encode(s), a, r))
            if done:
                break
            s = ns
        agent.update_episode(episode)
    # Should be able to finish at least sometimes after training
    successes = 0
    for _ in range(20):
        s = env.reset()
        for _ in range(20):
            a = agent.select_action(encode(s))
            s, r, done = env.step(a)
            if done:
                successes += 1
                break
    assert successes > 0


# --- DQN ---
def test_dqn_runs_on_small_env():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), random_state=0)

    def encode(s):
        v = np.zeros(env.n_states)
        v[s] = 1.0
        return v

    agent = DQNAgent(n_features=env.n_states, n_actions=env.n_actions,
                     hidden_sizes=(16,), lr=0.01, gamma=0.99,
                     epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1,
                     batch_size=16, memory_size=500, random_state=0)
    for _ in range(300):
        s = env.reset()
        for _ in range(30):
            a = agent.select_action(encode(s))
            ns, r, done = env.step(a)
            agent.store(encode(s), a, r, encode(ns), done)
            agent.update()
            if done:
                break
            s = ns
    # Some greedy rollouts
    agent.epsilon = 0.0
    wins = 0
    for _ in range(10):
        s = env.reset()
        for _ in range(20):
            a = agent.select_action(encode(s))
            s, r, done = env.step(a)
            if done:
                wins += 1
                break
    assert wins >= 1


# --- Double Q-Learning ---
def test_double_q_solves_grid():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), random_state=0)
    agent = DoubleQLearningAgent(env.n_states, env.n_actions, alpha=0.5, gamma=0.9,
                                 epsilon=0.3, random_state=0)
    for _ in range(500):
        s = env.reset()
        for _ in range(50):
            a = agent.select_action(s)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns, done)
            if done:
                break
            s = ns
    s = env.reset()
    for _ in range(20):
        a = int(np.argmax(agent.q_table_[s]))
        s, r, done = env.step(a)
        if done:
            break
    assert done


# --- Runner utilities ---
def test_run_episode_and_train():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), random_state=0)
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.5, gamma=0.9, epsilon=0.3, random_state=0)
    rewards = train_tabular(env, agent, episodes=200, max_steps=30)
    assert len(rewards) == 200
    avg = evaluate(env, agent, episodes=50, max_steps=30)
    assert avg > 0.5  # should solve grid


def test_sarsa_via_runner():
    env = GridWorld(rows=3, cols=3, start=(0, 0), goal=(2, 2), random_state=0)
    agent = SARSAAgent(env.n_states, env.n_actions, alpha=0.5, gamma=0.9, epsilon=0.3, random_state=0)
    for _ in range(500):
        run_episode(env, agent, max_steps=30)
    avg = evaluate(env, agent, episodes=50, max_steps=30)
    assert avg > 0.5
