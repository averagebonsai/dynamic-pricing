"""
rl_algorithms.py
================
Function-only reinforcement learning algorithms for ride-hailing experiments.

This module intentionally contains only generic RL algorithms.
It does not include data loading, graph construction, plotting, or a main script.

Expected single-agent environment interface
-------------------------------------------
state = env.reset()
next_state, reward = env.step(action)

Also supports Gym-style returns:
next_state, reward, done, info = env.step(action)
next_state, reward, terminated, truncated, info = env.step(action)

Expected multi-agent environment interface
------------------------------------------
state = env.reset()
next_state, rewards = env.step(actions)

where:
- actions has shape (n_agents,)
- rewards has shape (n_agents,)

All states are assumed to be discrete integer IDs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


def unpack_step(result: Tuple[Any, ...]) -> Tuple[Any, Any, bool, Dict[str, Any]]:
    """
    Normalize different environment step return formats.

    Supported formats:
    - (next_state, reward)
    - (next_state, reward, done, info)
    - (next_state, reward, terminated, truncated, info)
    """
    if len(result) == 2:
        next_state, reward = result
        return next_state, reward, False, {}

    if len(result) == 4:
        next_state, reward, done, info = result
        return next_state, reward, bool(done), dict(info or {})

    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
        return next_state, reward, done, dict(info or {})

    raise ValueError("Unsupported env.step(...) return format.")


def select_action(
    q_values: Sequence[float],
    eps: float = 0.1,
    policy: str = "epsilon_greedy",
    temperature: float = 1.0,
) -> int:
    """
    Select an action from Q-values.

    Parameters
    ----------
    q_values:
        Action values for the current state.
    eps:
        Exploration probability for epsilon-greedy.
    policy:
        One of: "epsilon_greedy", "softmax", "greedy", "random".
    temperature:
        Softmax temperature. Larger values produce more exploration.
    """
    q = np.asarray(q_values, dtype=float)
    n_actions = len(q)

    if n_actions == 0:
        raise ValueError("q_values must contain at least one action value.")

    if policy == "random":
        return int(np.random.randint(n_actions))

    if policy == "greedy":
        return int(np.argmax(q))

    if policy == "epsilon_greedy":
        if np.random.rand() < eps:
            return int(np.random.randint(n_actions))
        return int(np.argmax(q))

    if policy == "softmax":
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        z = q / temperature
        z = z - np.max(z)
        probs = np.exp(z) / np.exp(z).sum()
        return int(np.random.choice(n_actions, p=probs))

    raise ValueError(f"Unknown policy: {policy}")


def q_learning(
    env: Any,
    n_states: int,
    n_actions: int,
    episodes: int = 1000,
    steps_per_episode: int = 100,
    alpha: float = 0.05,
    discount: float = 0.95,
    eps: float = 0.1,
    policy: str = "epsilon_greedy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tabular Q-learning for a single agent.

    Update rule:
    Q(s,a) <- Q(s,a) + alpha * [r + discount * max_a' Q(s',a') - Q(s,a)]
    """
    q_table = np.zeros((n_states, n_actions), dtype=float)
    episode_rewards: List[float] = []

    for _ in range(episodes):
        state = int(env.reset())
        total_reward = 0.0

        for _ in range(steps_per_episode):
            action = select_action(q_table[state], eps=eps, policy=policy)
            next_state, reward, done, _ = unpack_step(env.step(action))
            next_state = int(next_state)
            reward = float(reward)

            target = reward + discount * np.max(q_table[next_state])
            q_table[state, action] += alpha * (target - q_table[state, action])

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

    return q_table, np.asarray(episode_rewards)


def sarsa(
    env: Any,
    n_states: int,
    n_actions: int,
    episodes: int = 1000,
    steps_per_episode: int = 100,
    alpha: float = 0.05,
    discount: float = 0.95,
    eps: float = 0.1,
    policy: str = "epsilon_greedy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tabular SARSA for a single agent.

    SARSA is on-policy: it updates using the next action actually selected by
    the behavior policy.
    """
    q_table = np.zeros((n_states, n_actions), dtype=float)
    episode_rewards: List[float] = []

    for _ in range(episodes):
        state = int(env.reset())
        action = select_action(q_table[state], eps=eps, policy=policy)
        total_reward = 0.0

        for _ in range(steps_per_episode):
            next_state, reward, done, _ = unpack_step(env.step(action))
            next_state = int(next_state)
            reward = float(reward)
            next_action = select_action(q_table[next_state], eps=eps, policy=policy)

            target = reward + discount * q_table[next_state, next_action]
            q_table[state, action] += alpha * (target - q_table[state, action])

            total_reward += reward
            state, action = next_state, next_action

            if done:
                break

        episode_rewards.append(total_reward)

    return q_table, np.asarray(episode_rewards)


def double_q_learning(
    env: Any,
    n_states: int,
    n_actions: int,
    episodes: int = 1000,
    steps_per_episode: int = 100,
    alpha: float = 0.05,
    discount: float = 0.95,
    eps: float = 0.1,
    policy: str = "epsilon_greedy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Double Q-learning for a single agent.

    This reduces overestimation bias by maintaining two independent Q-tables.
    """
    q1 = np.zeros((n_states, n_actions), dtype=float)
    q2 = np.zeros((n_states, n_actions), dtype=float)
    episode_rewards: List[float] = []

    for _ in range(episodes):
        state = int(env.reset())
        total_reward = 0.0

        for _ in range(steps_per_episode):
            action = select_action(q1[state] + q2[state], eps=eps, policy=policy)
            next_state, reward, done, _ = unpack_step(env.step(action))
            next_state = int(next_state)
            reward = float(reward)

            if np.random.rand() < 0.5:
                best_next = int(np.argmax(q1[next_state]))
                target = reward + discount * q2[next_state, best_next]
                q1[state, action] += alpha * (target - q1[state, action])
            else:
                best_next = int(np.argmax(q2[next_state]))
                target = reward + discount * q1[next_state, best_next]
                q2[state, action] += alpha * (target - q2[state, action])

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

    return q1, q2, np.asarray(episode_rewards)


def independent_q_learning(
    env: Any,
    n_agents: int,
    n_states: int,
    n_actions: int,
    episodes: int = 1000,
    steps_per_episode: int = 100,
    alpha: float = 0.05,
    discount: float = 0.95,
    eps: float = 0.1,
    policy: str = "epsilon_greedy",
    reward_transformer: Optional[Callable[[Any, float, int], float]] = None,
    callback: Optional[Callable[[int, np.ndarray, float, np.ndarray], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Independent Q-learning for multi-agent reinforcement learning.

    Each agent has its own Q-table and treats the other agents as part of the
    environment. This is a simple baseline for oligopoly experiments.
    """
    q_tables = np.zeros((n_agents, n_states, n_actions), dtype=float)
    episode_rewards: List[float] = []

    for ep in range(episodes):
        states = env.reset()
        if np.isscalar(states) or (isinstance(states, np.ndarray) and states.ndim == 0):
            states = np.full(n_agents, int(states), dtype=int)
        else:
            states = np.asarray(states, dtype=int)

        total_reward = 0.0
        actions_in_episode = []

        for step in range(steps_per_episode):
            actions = np.asarray([
                select_action(q_tables[i, states[i]], eps=eps, policy=policy)
                for i in range(n_agents)
            ], dtype=int)
            
            actions_in_episode.append(actions.copy())

            next_states, rewards, done, _ = unpack_step(env.step(actions))
            
            if np.isscalar(next_states) or (isinstance(next_states, np.ndarray) and next_states.ndim == 0):
                next_states = np.full(n_agents, int(next_states), dtype=int)
            else:
                next_states = np.asarray(next_states, dtype=int)
                
            rewards = np.asarray(rewards, dtype=float)

            for i in range(n_agents):
                action = actions[i]
                r = rewards[i]
                
                if reward_transformer is not None:
                    scaled_r = reward_transformer(env, r, i)
                elif hasattr(env, "taxis"):
                    n_taxis = env.taxis[i]
                    if n_taxis > 0:
                        avg_fare = r / n_taxis
                        scaled_r = np.clip(avg_fare / 100.0, 0.0, 1.0)
                    else:
                        scaled_r = 0.0
                else:
                    scaled_r = r
                    
                target = scaled_r + discount * np.max(q_tables[i, next_states[i]])
                q_tables[i, states[i], action] += alpha * (target - q_tables[i, states[i], action])

            total_reward += float(np.sum(rewards))
            states = next_states

            if done:
                break

        episode_rewards.append(total_reward)

        if callback is not None:
            callback(ep, q_tables, total_reward, np.asarray(actions_in_episode))

    return q_tables, np.asarray(episode_rewards)

class Exp3Agent:
    """
    Independent EXP3 Bandit Agent running on each pickup zone to choose surge prices.
    """
    def __init__(self, n_arms: int, n_actions: int = 4, gamma: float = 0.1, eta: float = 0.1):
        self.n_arms = n_arms
        self.n_actions = n_actions
        self.gamma = gamma
        self.eta = eta

        # Log weights initialization
        self.log_weights = np.zeros((n_arms, n_actions), dtype=float)
        self.probs = np.ones((n_arms, n_actions), dtype=float) / n_actions
        self._update_probs()

    def _update_probs(self):
        for arm in range(self.n_arms):
            log_w = self.log_weights[arm]
            w = np.exp(log_w - np.max(log_w)) # stable softmax
            sum_w = np.sum(w)
            if sum_w > 0:
                p_weights = w / sum_w
            else:
                p_weights = np.ones(self.n_actions) / self.n_actions

            # EXP3 probability distribution mixture
            self.probs[arm] = (1.0 - self.gamma) * p_weights + self.gamma / self.n_actions

    def select_actions(self) -> np.ndarray:
        actions = np.zeros(self.n_arms, dtype=int)
        for arm in range(self.n_arms):
            actions[arm] = np.random.choice(self.n_actions, p=self.probs[arm])
        return actions

    def update(self, actions: np.ndarray, rewards: np.ndarray):
        # actions and rewards have shape (n_arms,)
        for arm in range(self.n_arms):
            a = actions[arm]
            p = self.probs[arm, a]
            r = rewards[arm] # scaled reward in [0, 1]

            # EXP3 update rule
            estimated_reward = r / p
            self.log_weights[arm, a] += self.eta * estimated_reward

        self._update_probs()
