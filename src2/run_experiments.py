"""Supported command-line entry point for modular monopoly and oligopoly runs.

Examples:
    python -m src2.run_experiments --episodes 70 --steps 120
    python -m src2.run_experiments --prepare-assets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .assets import load_simulation_assets, prepare_assets
from .config import LearningConfig, SimulationConfig, parse_fleet_sizes
from .experiments import run_monopoly, run_oligopoly
from .reporting import plot_episode_revenue, save_episode_results, save_multiplier_summary


def parse_args() -> argparse.Namespace:
    """Keep runtime, economic, and learning parameters explicit at the boundary."""
    parser = argparse.ArgumentParser(description="Run modular dynamic-pricing monopoly and oligopoly experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/src2"))
    parser.add_argument("--prepare-assets", action="store_true", help="Prepare demand/fare files and train destination predictor, then exit.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--taxis", type=int, nargs="+", default=[5000])
    parser.add_argument("--n-states", type=int, default=10)
    parser.add_argument("--bin-size", type=int, default=10)
    parser.add_argument("--theta", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.2, help="Q-learning learning rate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Q-learning exploration probability")
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--exp3-gamma", type=float, default=0.1)
    parser.add_argument("--exp3-eta", type=float, default=0.1)
    parser.add_argument("--reward-normalizer", type=float, default=270.0)
    return parser.parse_args()


def main() -> None:
    """Prepare assets or execute both market structures and save comparable results."""
    args = parse_args()
    if args.prepare_assets:
        prepare_assets(args.data_dir, theta=args.theta)
        return
    simulation = SimulationConfig(
        fleet_sizes=parse_fleet_sizes(args.taxis),
        steps_per_episode=args.steps,
        n_inventory_states=args.n_states,
        inventory_bin_size=args.bin_size,
        price_sensitivity=args.theta,
    )
    learning = LearningConfig(
        episodes=args.episodes,
        discount=args.discount,
        q_learning_rate=args.alpha,
        q_epsilon=args.epsilon,
        exp3_exploration=args.exp3_gamma,
        exp3_learning_rate=args.exp3_eta,
        reward_normalizer=args.reward_normalizer,
        seed=args.seed,
    )
    assets = load_simulation_assets(args.data_dir)
    monopoly = run_monopoly(assets, simulation, learning)
    oligopoly = run_oligopoly(assets, simulation, learning)
    for market_name, result in (("monopoly", monopoly), ("oligopoly", oligopoly)):
        results_path = save_episode_results(result.records, args.output_dir, market_name)
        multiplier_path = save_multiplier_summary(result.multiplier_history, assets.zone_ids, args.output_dir, market_name)
        plot_path = plot_episode_revenue(result.records, args.output_dir, market_name)
        print(f"{market_name}: results={results_path}, multipliers={multiplier_path}, plot={plot_path}")


if __name__ == "__main__":
    main()
