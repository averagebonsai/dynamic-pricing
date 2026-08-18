"""Composable runners for the supported monopoly and oligopoly experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algorithms import ZoneExp3, ZoneQLearner
from .assets import SimulationAssets
from .config import LearningConfig, SimulationConfig
from .simulation import TaxiMarketEnv


@dataclass
class ExperimentResult:
    """Raw episode records and price choices for reporting or further analysis."""

    records: list[dict]
    multiplier_history: np.ndarray


def run_monopoly(assets: SimulationAssets, simulation: SimulationConfig, learning: LearningConfig) -> ExperimentResult:
    """Train independent zone-level tabular Q-learners in a one-firm market."""
    simulation.validate(n_firms=1)
    learning.validate()
    rng = np.random.default_rng(learning.seed)
    environment = TaxiMarketEnv(assets, simulation, n_firms=1, rng=rng)
    learner = ZoneQLearner(
        n_zones=assets.n_zones,
        n_states=simulation.n_inventory_states,
        n_actions=len(simulation.price_actions),
        learning_rate=learning.q_learning_rate,
        discount=learning.discount,
        epsilon=learning.q_epsilon,
        rng=np.random.default_rng(learning.seed + 1),
    )
    action_values = np.asarray(simulation.price_actions, dtype=float)
    records: list[dict] = []
    multiplier_history: list[np.ndarray] = []
    for episode in range(learning.episodes):
        states = environment.reset()[0]
        episode_revenue = 0.0
        episode_multipliers: list[np.ndarray] = []
        for _ in range(simulation.steps_per_episode):
            action_indexes = learner.select_actions(states)
            multipliers = action_values[action_indexes]
            next_all_states, revenue, done, _ = environment.step(multipliers[None, :])
            next_states = next_all_states[0]
            # Q-values receive bounded revenue, while result records preserve
            # economically meaningful dollar revenue.
            scaled_revenue = np.clip(revenue[0] / learning.reward_normalizer, 0.0, 1.0)
            learner.update(states, action_indexes, scaled_revenue, next_states, done=done)
            episode_revenue += float(revenue.sum())
            episode_multipliers.append(multipliers[None, :])
            states = next_states
            if done:
                break
        multiplier_history.append(np.asarray(episode_multipliers))
        records.append({"episode": episode, "firm_id": 0, "revenue": episode_revenue})
    return ExperimentResult(records, np.asarray(multiplier_history))


def run_oligopoly(assets: SimulationAssets, simulation: SimulationConfig, learning: LearningConfig, n_firms: int = 4) -> ExperimentResult:
    """Train one independent zone-level EXP3 policy collection per platform."""
    simulation.validate(n_firms=n_firms)
    learning.validate()
    environment = TaxiMarketEnv(assets, simulation, n_firms=n_firms, rng=np.random.default_rng(learning.seed + 10_000))
    agents = [
        ZoneExp3(
            n_zones=assets.n_zones,
            n_actions=len(simulation.price_actions),  # Includes the 2.0 multiplier.
            exploration=learning.exp3_exploration,
            learning_rate=learning.exp3_learning_rate,
            reward_normalizer=learning.reward_normalizer,
            rng=np.random.default_rng(learning.seed + 20_000 + firm),
        )
        for firm in range(n_firms)
    ]
    action_values = np.asarray(simulation.price_actions, dtype=float)
    records: list[dict] = []
    multiplier_history: list[np.ndarray] = []
    for episode in range(learning.episodes):
        environment.reset()
        episode_revenue = np.zeros(n_firms, dtype=float)
        episode_multipliers: list[np.ndarray] = []
        for _ in range(simulation.steps_per_episode):
            actions = np.vstack([agent.select_actions() for agent in agents])
            multipliers = action_values[actions]
            _, revenue, done, _ = environment.step(multipliers)
            for firm, agent in enumerate(agents):
                agent.update(actions[firm], revenue[firm])
            episode_revenue += revenue.sum(axis=1)
            episode_multipliers.append(multipliers)
            if done:
                break
        multiplier_history.append(np.asarray(episode_multipliers))
        records.extend(
            {"episode": episode, "firm_id": firm, "revenue": float(episode_revenue[firm])}
            for firm in range(n_firms)
        )
    return ExperimentResult(records, np.asarray(multiplier_history))
