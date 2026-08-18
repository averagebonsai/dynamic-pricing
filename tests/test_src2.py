"""Focused checks for the supported modular simulation pipeline."""

from pathlib import Path

import numpy as np

from src2.algorithms import ZoneQLearner
from src2.assets import load_simulation_assets
from src2.config import LearningConfig, SimulationConfig
from src2.experiments import run_monopoly
from src2.simulation import TaxiMarketEnv


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def test_zone_qlearner_does_not_bootstrap_terminal_transition() -> None:
    """A terminal Q-learning target is reward only, never reward plus future value."""
    learner = ZoneQLearner(
        n_zones=2,
        n_states=2,
        n_actions=2,
        learning_rate=1.0,
        discount=0.9,
        epsilon=0.0,
        rng=np.random.default_rng(42),
    )
    learner.q_values[:, 1] = np.array([[3.0, 7.0], [11.0, 5.0]])
    states = np.array([0, 0])
    actions = np.array([0, 1])
    rewards = np.array([2.0, 4.0])
    next_states = np.array([1, 1])

    learner.update(states, actions, rewards, next_states, done=True)

    np.testing.assert_allclose(learner.q_values[np.arange(2), states, actions], rewards)


def test_zone_qlearner_bootstraps_non_terminal_transition() -> None:
    """A continuing transition retains the discounted maximum future value."""
    learner = ZoneQLearner(
        n_zones=1,
        n_states=2,
        n_actions=2,
        learning_rate=1.0,
        discount=0.9,
        epsilon=0.0,
        rng=np.random.default_rng(42),
    )
    learner.q_values[0, 1] = np.array([3.0, 7.0])

    learner.update(np.array([0]), np.array([0]), np.array([2.0]), np.array([1]))

    np.testing.assert_allclose(learner.q_values[0, 0, 0], 2.0 + 0.9 * 7.0)


def test_prepared_assets_and_inventory_states_match_runtime_contract() -> None:
    """The committed prepared data exposes 262 zones and bounded inventory states."""
    assets = load_simulation_assets(DATA_DIR)
    config = SimulationConfig(fleet_sizes=(500,), n_inventory_states=10, inventory_bin_size=10)
    environment = TaxiMarketEnv(assets, config, n_firms=1, rng=np.random.default_rng(42))

    states = environment.reset()

    assert assets.n_zones == 262
    assert states.shape == (1, 262)
    assert np.all((0 <= states) & (states < config.n_inventory_states))


def test_monopoly_runner_smoke() -> None:
    """A minimal supported Q-learning experiment completes with expected shapes."""
    assets = load_simulation_assets(DATA_DIR)
    result = run_monopoly(
        assets,
        SimulationConfig(fleet_sizes=(500,), steps_per_episode=1, n_inventory_states=10, inventory_bin_size=10),
        LearningConfig(episodes=1, seed=42),
    )

    assert len(result.records) == 1
    assert result.multiplier_history.shape == (1, 1, 1, 262)
    assert np.isfinite(result.records[0]["revenue"])
