"""agent.py - Tabular epsilon-greedy Q-learning for the Smart Cafe (backward compatible).

    train(env, ...)                           -> (Q, reward_curve)          # assignment contract
    train(env, ..., return_diagnostics=True)  -> (Q, reward_curve, diag)   # optional extras
    policy_from_Q(Q)                          -> best order for each stock level

The diagnostics only COUNT what happens; they never change action sampling, epsilon decay,
rewards or the Q-update.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np


def epsilon_at(k: int, eps_start: float = 1.0, eps_min: float = 0.05,
               eps_decay: float = 0.9998) -> float:
    """Epsilon used during episode k (0-based): max(eps_min, eps_start * eps_decay**k)."""
    return max(eps_min, eps_start * eps_decay ** k)


def floor_episode(eps_start: float = 1.0, eps_min: float = 0.05,
                  eps_decay: float = 0.9998) -> Optional[int]:
    """First 0-based episode index whose epsilon sits exactly on the floor."""
    if eps_start <= eps_min:
        return 0
    if not 0 < eps_decay < 1:
        return None
    return math.ceil(math.log(eps_min / eps_start) / math.log(eps_decay))


def train(env, episodes: int = 15000, alpha: float = 0.15, gamma: float = 0.95,
          eps_start: float = 1.0, eps_min: float = 0.05, eps_decay: float = 0.9998,
          seed: Optional[int] = None, return_diagnostics: bool = False):
    n_states = n_actions = env.max_stock + 1       # stock 0..15, order 0..15 -> 16 x 16
    Q = np.zeros((n_states, n_actions))            # starts knowing NOTHING
    reward_curve = []
    if seed is None:
        g = np.random.default_rng()
    else:  # two independent child streams: environment demand vs. exploration
        env_ss, agent_ss = np.random.SeedSequence(seed).spawn(2)
        env.rng = np.random.default_rng(env_ss)
        g = np.random.default_rng(agent_ss)

    eps_hist = np.empty(episodes)
    rand_n = np.zeros(episodes, dtype=int)         # random branch taken (logged, not inferred)
    nongreedy_n = np.zeros(episodes, dtype=int)    # random branch AND action != greedy
    visits = np.zeros((n_states, n_actions), dtype=int)
    eps = eps_start

    for ep in range(episodes):
        state = env.reset()
        done, total = False, 0.0
        eps_hist[ep] = eps
        while not done:
            greedy = int(np.argmax(Q[state]))      # exploit choice (uses no randomness)
            if g.random() < eps:                   # explore: random order 0..15
                action = int(g.integers(n_actions))
                rand_n[ep] += 1
                nongreedy_n[ep] += int(action != greedy)
            else:                                  # exploit: best-known order
                action = greedy
            reward, next_state, done, _ = env.step(action)
            # Q-learning update: Q[s,a] += alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            visits[state, action] += 1
            state = next_state
            total += reward
        reward_curve.append(total)
        eps = max(eps_min, eps * eps_decay)        # decay once per episode (assignment schedule)

    if not return_diagnostics:
        return Q, reward_curve
    steps = int(env.days)
    diag: Dict[str, Any] = dict(
        eps=eps_hist, random_branch=rand_n, greedy_branch=steps - rand_n, nongreedy=nongreedy_n,
        sa_visits=visits, state_visits=visits.sum(axis=1),
        config=dict(episodes=episodes, alpha=alpha, gamma=gamma, eps_start=eps_start,
                    eps_min=eps_min, eps_decay=eps_decay, seed=seed))
    return Q, reward_curve, diag


def policy_from_Q(Q: np.ndarray) -> np.ndarray:
    """Greedy policy: best order per stock level. Ties (e.g. never-visited all-zero rows) -> 0."""
    return np.argmax(Q, axis=1)
