from __future__ import annotations

import numpy as np


def run_episode(env, agent, max_steps=200, train=True):
    """Run one episode and return total reward.

    Automatically detects agent type:
    - Tabular per-step: QLearning, DoubleQ (update needs next_state only)
    - SARSA: update needs next_action
    - Episodic: REINFORCE (update called once at end)
    - DQN: has .store() method
    """
    # Detect agent type from update signature
    import inspect
    try:
        sig = inspect.signature(agent.update)
        params = list(sig.parameters.keys())
        needs_next_action = 'next_action' in params
    except (ValueError, TypeError):
        needs_next_action = False

    # Episodic agents
    if hasattr(agent, 'update_episode'):
        episode = []
        state = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            total += reward
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        if train:
            agent.update_episode(episode)
        return total

    # Per-step agents
    state = env.reset()
    total = 0.0
    action = agent.select_action(state)
    for _ in range(max_steps):
        next_state, reward, done = env.step(action)
        total += reward
        next_action = agent.select_action(next_state)
        if needs_next_action:
            agent.update(state, action, reward, next_state, next_action, done)
        else:
            agent.update(state, action, reward, next_state, done)
        if done:
            break
        state, action = next_state, next_action
    return total


def run_dqn_episode(env, agent, max_steps=200):
    """Run one DQN episode with experience replay."""
    state = env.reset()
    total = 0.0
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total += reward
        agent.store(state, action, reward, next_state, done)
        agent.update()
        if done:
            break
        state = next_state
    return total


def train_tabular(env, agent, episodes=1000, max_steps=200, verbose=False):
    """Train a tabular agent (Q/SARSA/DoubleQ) for N episodes."""
    rewards = []
    for ep in range(episodes):
        total = run_episode(env, agent, max_steps=max_steps)
        rewards.append(total)
        if verbose and (ep + 1) % max(1, episodes // 10) == 0:
            print(f"  ep {ep + 1}/{episodes}  avg_reward={np.mean(rewards[-100:]):.2f}")
    return rewards


def evaluate(env, agent, episodes=100, max_steps=200):
    """Evaluate an agent greedily (no exploration)."""
    old_eps = getattr(agent, 'epsilon', None)
    if old_eps is not None:
        agent.epsilon = 0.0
    totals = []
    for _ in range(episodes):
        if hasattr(agent, 'store'):
            totals.append(run_dqn_episode(env, agent, max_steps=max_steps))
        else:
            totals.append(run_episode(env, agent, max_steps=max_steps, train=False))
    if old_eps is not None:
        agent.epsilon = old_eps
    return float(np.mean(totals))
