"""
rmab_algorithms.py
==================
Function-only algorithms for Restless Multi-Armed Bandit (RMAB) experiments.

RMAB does not necessarily require reinforcement learning.
If the transition probabilities and rewards are known, RMAB policies can be
computed using model-based methods such as value iteration and index policies.
If the model is unknown, reinforcement learning can be used to learn values or
indices from interaction with an environment.

Expected multi-arm environment interface for learning-based RMAB
----------------------------------------------------------------
states = env.reset()
next_states, rewards = env.step(actions)

where:
- states has shape (n_arms,)
- actions has shape (n_arms,), with 1 = active and 0 = passive
- rewards has shape (n_arms,)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


def unpack_step(result: Tuple[Any, ...]) -> Tuple[Any, Any, bool, Dict[str, Any]]:
    """
    Normalize different environment step return formats.
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


def rmab_index_policy(indices: Sequence[float], budget: int) -> np.ndarray:
    """
    Activate the arms with the highest index values.

    Parameters
    ----------
    indices:
        Priority score for each arm.
    budget:
        Maximum number of arms that can be active at the same time.

    Returns
    -------
    actions:
        Binary action vector. 1 = active, 0 = passive.
    """
    indices = np.asarray(indices, dtype=float)
    n_arms = len(indices)
    budget = min(int(budget), n_arms)

    actions = np.zeros(n_arms, dtype=int)
    if budget <= 0:
        return actions

    active_arms = np.argsort(indices)[-budget:]
    actions[active_arms] = 1
    return actions


def model_based_q_value_iteration(
    transition: np.ndarray,
    reward: np.ndarray,
    discount: float = 0.95,
    subsidy: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> np.ndarray:
    """
    Model-based Q-value iteration for a single restless arm.

    Parameters
    ----------
    transition:
        Transition matrix with shape (2, n_states, n_states).
        transition[a, s, s_next] = P(s_next | s, a).
    reward:
        Reward matrix with shape (n_states, 2).
        reward[s, a] = immediate reward for state s and action a.
    discount:
        Discount factor.
    subsidy:
        Passive-action subsidy added when action == 0.
    tol:
        Convergence tolerance.
    max_iter:
        Maximum number of value iteration steps.

    Returns
    -------
    q_values:
        Q-table with shape (n_states, 2).
    """
    transition = np.asarray(transition, dtype=float)
    reward = np.asarray(reward, dtype=float)
    n_states = reward.shape[0]
    q_values = np.zeros((n_states, 2), dtype=float)

    for _ in range(max_iter):
        old_q = q_values.copy()
        value = np.max(old_q, axis=1)

        for s in range(n_states):
            for a in range(2):
                passive_subsidy = subsidy if a == 0 else 0.0
                q_values[s, a] = (
                    reward[s, a]
                    + passive_subsidy
                    + discount * np.dot(transition[a, s], value)
                )

        if np.max(np.abs(q_values - old_q)) < tol:
            break

    return q_values


def compute_model_based_whittle_indices(
    transition: np.ndarray,
    reward: np.ndarray,
    discount: float = 0.95,
    lambda_low: float = -100.0,
    lambda_high: float = 100.0,
    tol: float = 1e-6,
    max_bisection_iter: int = 80,
) -> np.ndarray:
    """
    Compute approximate Whittle indices for a single-arm model by bisection.

    This assumes the arm is indexable. If indexability does not hold, the output
    should be interpreted only as a numerical heuristic.

    The index for state s is the subsidy lambda that makes active and passive
    actions equally valuable:
        Q_lambda(s, active) - Q_lambda(s, passive) = 0
    """
    reward = np.asarray(reward, dtype=float)
    n_states = reward.shape[0]
    indices = np.zeros(n_states, dtype=float)

    for s in range(n_states):
        low, high = lambda_low, lambda_high

        for _ in range(max_bisection_iter):
            mid = 0.5 * (low + high)
            q_values = model_based_q_value_iteration(
                transition=transition,
                reward=reward,
                discount=discount,
                subsidy=mid,
                tol=tol,
            )
            diff = q_values[s, 1] - q_values[s, 0]

            if diff > 0:
                low = mid
            else:
                high = mid

        indices[s] = 0.5 * (low + high)

    return indices


def rmab_q_learning(
    env: Any,
    n_arms: int,
    n_states: int,
    budget: int,
    episodes: int = 1000,
    steps_per_episode: int = 100,
    alpha: float = 0.05,
    discount: float = 0.95,
    eps: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple learning-based RMAB algorithm using per-arm Q-learning.

    This is not a Whittle-index algorithm. It learns Q[arm, state, action] and
    uses the active-passive Q gap as a priority score:
        index_like_score = Q(arm, state, active) - Q(arm, state, passive)

    At each step, the top `budget` arms are activated.
    """
    n_actions = 2
    q_values = np.zeros((n_arms, n_states, n_actions), dtype=float)
    episode_rewards: List[float] = []

    for _ in range(episodes):
        states = np.asarray(env.reset(), dtype=int)
        total_reward = 0.0

        for _ in range(steps_per_episode):
            priority_scores = q_values[np.arange(n_arms), states, 1] - q_values[np.arange(n_arms), states, 0]

            if np.random.rand() < eps:
                actions = np.zeros(n_arms, dtype=int)
                active = np.random.choice(n_arms, size=min(budget, n_arms), replace=False)
                actions[active] = 1
            else:
                actions = rmab_index_policy(priority_scores, budget)

            next_states, rewards, done, _ = unpack_step(env.step(actions))
            next_states = np.asarray(next_states, dtype=int)
            rewards = np.asarray(rewards, dtype=float)

            for arm in range(n_arms):
                s = states[arm]
                a = actions[arm]
                ns = next_states[arm]

                target = rewards[arm] + discount * np.max(q_values[arm, ns])
                q_values[arm, s, a] += alpha * (target - q_values[arm, s, a])

            total_reward += float(np.sum(rewards))
            states = next_states

            if done:
                break

        episode_rewards.append(total_reward)

    return q_values, np.asarray(episode_rewards)


def whittle_index_learning(
    env: Any,
    n_states: int,
    outer_iterations: int = 500,
    inner_iterations: int = 1000,
    alpha: float = 0.05,
    eta: float = 0.01,
    discount: float = 0.95,
    eps: float = 0.1,
    tolerance: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Model-free Whittle index learning for a single restless arm.

    This algorithm uses an inner Q-learning loop and an outer index update loop.
    It is included here as an RMAB-specific learning method, separate from the
    generic RL algorithms in rl_algorithms.py.

    Expected environment:
    - env.reset() -> state
    - env.step(action) -> next_state, reward

    Actions:
    - 0 = passive
    - 1 = active
    """
    n_actions = 2
    q_values = np.zeros((n_states, n_actions, n_states), dtype=float)
    indices = np.zeros(n_states, dtype=float)
    errors: List[float] = []

    for _ in range(outer_iterations):
        for threshold_state in range(n_states):
            q_slice = q_values[:, :, threshold_state].copy()
            state = int(env.reset())

            for _ in range(inner_iterations):
                if np.random.rand() < eps:
                    action = int(np.random.randint(n_actions))
                else:
                    action = int(np.argmax(q_slice[state]))

                next_state, reward, done, _ = unpack_step(env.step(action))
                next_state = int(next_state)
                reward = float(reward)

                passive_subsidy = indices[threshold_state] if action == 0 else 0.0
                target = reward + passive_subsidy + discount * np.max(q_slice[next_state])
                q_slice[state, action] += alpha * (target - q_slice[state, action])

                state = int(env.reset()) if done else next_state

            q_values[:, :, threshold_state] = q_slice

        max_error = 0.0
        for threshold_state in range(n_states):
            diff = q_values[threshold_state, 1, threshold_state] - q_values[threshold_state, 0, threshold_state]
            indices[threshold_state] += eta * diff
            max_error = max(max_error, abs(diff))

        errors.append(max_error)
        if max_error < tolerance:
            break

    return indices, q_values, np.asarray(errors)
