"""Reusable, stateful learning agents for zone-level dynamic pricing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ZoneQLearner:
    """Independent tabular Q-learning, one state/action table per pickup zone.

    The zones are factorised deliberately: learning a joint policy over every
    zone's multiplier would require an infeasibly large action space.

    Corresponds to a 3D tensor of (252, 10, 5): 
    - 252 zones, 10 bins for number of taxis, 5 potential actions. 
    """

    n_zones: int
    n_states: int
    n_actions: int
    learning_rate: float
    discount: float
    epsilon: float
    rng: np.random.Generator
    q_values: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.q_values = np.zeros((self.n_zones, self.n_states, self.n_actions), dtype=float)

    def select_actions(self, states: np.ndarray) -> np.ndarray:
        """Choose one price-action index per zone using epsilon-greedy policy."""
        states = np.asarray(states, dtype=int)
        greedy = np.argmax(self.q_values[np.arange(self.n_zones), states], axis=1)
        explore = self.rng.random(self.n_zones) < self.epsilon
        random_actions = self.rng.integers(self.n_actions, size=self.n_zones)
        return np.where(explore, random_actions, greedy).astype(int)

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool = False,
    ) -> None:
        """Apply an independent Q-learning update for every zone.

        Terminal transitions have no future value, so their target is the
        immediate reward rather than a bootstrapped Q-value.
        """
        zones = np.arange(self.n_zones)
        rewards = np.asarray(rewards, dtype=float)
        future_values = self.q_values[zones, next_states].max(axis=1)
        targets = rewards if done else rewards + self.discount * future_values
        current = self.q_values[zones, states, actions]
        self.q_values[zones, states, actions] += self.learning_rate * (targets - current)


@dataclass
class ZoneExp3:
    """Numerically stable reward-form EXP3 with one bandit per pickup zone.

    Rewards must be raw non-negative revenue.  Scaling is owned here rather
    than scattered across runners, keeping experiments comparable.
    """

    n_zones: int
    n_actions: int
    exploration: float
    learning_rate: float
    reward_normalizer: float
    rng: np.random.Generator
    max_log_weight: float = 50.0
    max_estimated_reward: float = 50.0
    log_weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n_actions < 2:
            raise ValueError("EXP3 requires at least two actions.")
        self.log_weights = np.zeros((self.n_zones, self.n_actions), dtype=float)

    def reset(self) -> None:
        """Forget learned preferences while retaining this agent's settings/RNG."""
        self.log_weights.fill(0.0)

    def probabilities(self) -> np.ndarray:
        """Return exploration-mixed action probabilities for every zone."""
        centered = self.log_weights - self.log_weights.max(axis=1, keepdims=True)
        base_weights = np.exp(np.clip(centered, -self.max_log_weight, 0.0))
        base_probabilities = base_weights / np.maximum(base_weights.sum(axis=1, keepdims=True), 1e-12)
        probabilities = (1.0 - self.exploration) * base_probabilities + self.exploration / self.n_actions
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def select_actions(self) -> np.ndarray:
        """Sample the selected price-action index for each zone."""
        probabilities = self.probabilities()
        return np.asarray(
            [self.rng.choice(self.n_actions, p=probabilities[zone]) for zone in range(self.n_zones)],
            dtype=int,
        )

    def update(self, actions: np.ndarray, raw_rewards: np.ndarray) -> None:
        """Update selected arms using importance-weighted, normalized revenue."""
        actions = np.asarray(actions, dtype=int)
        rewards = np.nan_to_num(np.asarray(raw_rewards, dtype=float), nan=0.0, posinf=self.reward_normalizer, neginf=0.0)
        scaled_rewards = np.clip(rewards / self.reward_normalizer, 0.0, 1.0)
        probabilities = self.probabilities()
        zones = np.arange(self.n_zones)
        selected_probabilities = np.maximum(probabilities[zones, actions], 1e-12)
        estimates = np.clip(scaled_rewards / selected_probabilities, 0.0, self.max_estimated_reward)
        self.log_weights[zones, actions] += self.learning_rate * estimates

        # Shifting is mathematically neutral and bounds exponentiation later.
        self.log_weights -= self.log_weights.max(axis=1, keepdims=True)
        self.log_weights[:] = np.clip(self.log_weights, -self.max_log_weight, 0.0)

    def expected_action_values(self, action_values: np.ndarray) -> np.ndarray:
        """Return the expected multiplier/value per zone under current policy."""
        return self.probabilities() @ np.asarray(action_values, dtype=float)
