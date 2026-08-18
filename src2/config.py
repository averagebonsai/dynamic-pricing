"""Typed, validated configuration for the new simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# This is the single source of truth for pricing actions.  Algorithms only
# choose integer indexes; the environment always receives the actual values.
PRICE_ACTIONS: tuple[float, ...] = (1.0, 1.2, 1.5, 1.8, 2.0)


@dataclass(frozen=True)
class SimulationConfig:
    """Economic and operational assumptions shared by all market structures."""

    fleet_sizes: tuple[int, ...] = (5_000,)
    steps_per_episode: int = 120
    n_inventory_states: int = 10
    inventory_bin_size: int = 10
    price_sensitivity: float = 0.4
    start_day: int = 0
    start_hour: int = 0
    price_actions: tuple[float, ...] = PRICE_ACTIONS

    def validate(self, n_firms: int) -> None:
        """Reject configuration combinations that would silently change a run."""
        if n_firms < 1:
            raise ValueError("n_firms must be at least one.")
        if self.steps_per_episode < 1 or self.n_inventory_states < 1 or self.inventory_bin_size < 1:
            raise ValueError("Steps, inventory states, and bin size must be positive.")
        if self.price_sensitivity < 0:
            raise ValueError("price_sensitivity must be non-negative.")
        if len(self.price_actions) < 2 or any(multiplier <= 0 for multiplier in self.price_actions):
            raise ValueError("price_actions must contain at least two positive multipliers.")
        if len(self.fleet_sizes) not in (1, n_firms):
            raise ValueError("fleet_sizes must contain one total fleet or one value per firm.")
        if any(size < 0 for size in self.fleet_sizes):
            raise ValueError("Fleet sizes cannot be negative.")

    def resolved_fleet_sizes(self, n_firms: int) -> np.ndarray:
        """Return one fleet size per firm, preserving any explicit asymmetry."""
        self.validate(n_firms)
        if len(self.fleet_sizes) == n_firms:
            return np.asarray(self.fleet_sizes, dtype=int)
        total = self.fleet_sizes[0]
        if n_firms == 1:
            return np.asarray([total], dtype=int)
        # Retain the old project's convention: equal fleets, with any remainder
        # intentionally not assigned rather than hidden in an arbitrary firm.
        return np.full(n_firms, total // n_firms, dtype=int)


@dataclass(frozen=True)
class LearningConfig:
    """Hyperparameters used by the Q-learning and EXP3 runners."""

    episodes: int = 100
    discount: float = 0.95
    q_learning_rate: float = 0.2
    q_epsilon: float = 0.1
    exp3_exploration: float = 0.1
    exp3_learning_rate: float = 0.1
    reward_normalizer: float = 270.0
    seed: int = 42

    def validate(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be positive.")
        if not 0 <= self.discount <= 1 or not 0 <= self.q_epsilon <= 1:
            raise ValueError("discount and q_epsilon must be in [0, 1].")
        if not 0 <= self.exp3_exploration <= 1:
            raise ValueError("exp3_exploration must be in [0, 1].")
        if self.q_learning_rate <= 0 or self.exp3_learning_rate <= 0 or self.reward_normalizer <= 0:
            raise ValueError("Learning rates and reward_normalizer must be positive.")


def parse_fleet_sizes(values: Sequence[int]) -> tuple[int, ...]:
    """Convert CLI fleet arguments into the immutable configuration form."""
    return tuple(int(value) for value in values)
