#!/usr/bin/env python3
"""
courthoud_freeze_unfreeze_monopoly_reset.py
===========================================
One-file Courthoud-style freeze/unfreeze experiment with frozen platforms
resetting their memory upon unfreeze.

Main experiment flow
--------------------
1. Train four oligopoly platforms normally with zone-level EXP3.
2. Evaluate learned expected policies before freeze.
3. Freeze three platforms at their learned expected-policy multipliers.
4. Let one deviating platform continue learning without reset.
5. Wipe the memory/weights from the three frozen platforms.
6. Unfreeze them and let all four platforms learn again (with the three learning from scratch).
7. Plot Total Revenue with:
   - Platform 1 = blue
   - Platform 2 = orange
   - Platform 3 = green
   - Platform 4 = red
   - Deviator = solid line
   - Frozen / then unfrozen platforms = dashed lines
   - 1/4 Monopoly benchmark = solid black line
   - THE FREEZE = gray dashed vertical line
   - THE UNFREEZE = purple dotted vertical line
"""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "src"))

from destination import DestinationNNPredictor
from simulations import OligopolyTaxiEnv

# Needed when destination_predictor.pkl was saved while destination.py was run as __main__.
sys.modules["__main__"].DestinationNNPredictor = DestinationNNPredictor

MULTIPLIERS = np.array([1.0, 1.2, 1.5, 1.8, 2.0], dtype=float)
N_FIRMS = 4
PLATFORM_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}
FREEZE_COLOR = "0.45"
UNFREEZE_COLOR = "purple"
MONOPOLY_COLOR = "black"


class ZoneExp3Adapter:
    """Stable zone-level EXP3 agent using log weights."""

    def __init__(self, n_arms: int, gamma: float, eta: float, reward_normalizer: float, rng: np.random.Generator):
        self.n_arms = int(n_arms)
        self.n_actions = len(MULTIPLIERS)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.reward_normalizer = float(reward_normalizer)
        self.rng = rng
        self.log_weights = np.zeros((self.n_arms, self.n_actions), dtype=float)
        self.max_abs_log_weight = 50.0
        self.max_importance_weighted_reward = 50.0

    def probabilities(self) -> np.ndarray:
        z = self.log_weights - self.log_weights.max(axis=1, keepdims=True)
        exp_z = np.exp(np.clip(z, -self.max_abs_log_weight, 0.0))
        base = exp_z / np.maximum(exp_z.sum(axis=1, keepdims=True), 1e-12)
        probs = (1.0 - self.gamma) * base + self.gamma / self.n_actions
        probs = np.nan_to_num(probs, nan=1.0 / self.n_actions, posinf=1.0 / self.n_actions, neginf=1.0 / self.n_actions)
        probs = np.maximum(probs, 1e-12)
        return probs / probs.sum(axis=1, keepdims=True)

    def select_actions(self) -> np.ndarray:
        probs = self.probabilities()
        actions = np.empty(self.n_arms, dtype=int)
        for arm in range(self.n_arms):
            p = probs[arm]
            p = p / p.sum()
            actions[arm] = int(self.rng.choice(self.n_actions, p=p))
        return actions

    def update(self, actions: np.ndarray, rewards: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=self.reward_normalizer, neginf=0.0)
        probs = self.probabilities()
        scaled = np.clip(rewards / max(self.reward_normalizer, 1e-9), 0.0, 1.0)
        for arm in range(self.n_arms):
            action = int(actions[arm])
            p = max(float(probs[arm, action]), 1e-12)
            est_reward = float(np.clip(scaled[arm] / p, 0.0, self.max_importance_weighted_reward))
            self.log_weights[arm, action] += self.eta * est_reward
        self.log_weights -= self.log_weights.max(axis=1, keepdims=True)
        self.log_weights = np.clip(self.log_weights, -self.max_abs_log_weight, 0.0)

    def expected_multipliers(self) -> np.ndarray:
        return self.probabilities() @ MULTIPLIERS


def moving_average_same_length(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_inputs(data_dir: Path):
    latent_df = pl.read_csv(data_dir / "latent_mle_params.csv")
    graph_df = pl.read_csv(data_dir / "graph.csv")
    with open(data_dir / "destination_predictor.pkl", "rb") as f:
        predictor = pickle.load(f)

    do_to_pu = {}
    for do_val in predictor.unique_do:
        do_to_pu[do_val] = do_val if do_val in predictor.unique_pu else predictor.unique_pu[0]

    graph_dict = {}
    for row in graph_df.iter_rows(named=True):
        pu = int(row["PULocationID"])
        do = int(row["DOLocationID"])
        graph_dict.setdefault(pu, {})[do] = {
            "distance": float(row["baseline_distance"]),
            "fare": float(row["baseline_fare"]),
        }

    x = graph_df["baseline_distance"].to_numpy().reshape(-1, 1)
    y = graph_df["baseline_fare"].to_numpy()
    reg = LinearRegression().fit(x, y)
    fare_params = {"price_per_mile": float(reg.coef_[0]), "intercept": float(reg.intercept_)}
    return latent_df, graph_dict, predictor, do_to_pu, fare_params


def make_env(args, latent_df, graph_dict, predictor, do_to_pu, fare_params) -> OligopolyTaxiEnv:
    return OligopolyTaxiEnv(
        latent_df=latent_df,
        graph_dict=graph_dict,
        predictor=predictor,
        do_to_pu=do_to_pu,
        fleet_size=args.taxis,
        n_states=args.n_states,
        bin_size=args.bin_size,
        theta=args.theta,
        steps_per_episode=args.steps,
        start_day=args.start_day,
        start_hour=args.start_hour,
        price_per_mile=fare_params["price_per_mile"],
        intercept=fare_params["intercept"],
    )


def agents_to_expected_multipliers(agents: list[ZoneExp3Adapter]) -> np.ndarray:
    return np.vstack([agent.expected_multipliers() for agent in agents])


def train_baseline(args, latent_df, graph_dict, predictor, do_to_pu, fare_params) -> list[ZoneExp3Adapter]:
    env0 = make_env(args, latent_df, graph_dict, predictor, do_to_pu, fare_params)
    agents = [
        ZoneExp3Adapter(env0.n_arms, args.gamma, args.eta, args.reward_normalizer, np.random.default_rng(args.seed + 10_000 + firm))
        for firm in range(N_FIRMS)
    ]

    for ep in range(args.episodes):
        env = make_env(args, latent_df, graph_dict, predictor, do_to_pu, fare_params)
        env.reset()
        total_profit = np.zeros(N_FIRMS)
        total_mult = np.zeros(N_FIRMS)
        for _ in range(args.steps):
            actions_by_firm = [agent.select_actions() for agent in agents]
            multipliers = np.vstack([MULTIPLIERS[a] for a in actions_by_firm])
            _, rewards_by_zone, done, _ = env.step(multipliers)
            for firm in range(N_FIRMS):
                agents[firm].update(actions_by_firm[firm], rewards_by_zone[firm])
                total_profit[firm] += float(rewards_by_zone[firm].sum())
                total_mult[firm] += float(multipliers[firm].mean())
            if done:
                break
        if args.log_every and ((ep + 1) % args.log_every == 0 or ep == 0):
            print(f"baseline train {ep+1}/{args.episodes} | revenue={np.round(total_profit, 2).tolist()} | avg_mult={np.round(total_mult / args.steps, 3).tolist()}")
    return agents


def evaluate_before_freeze(args, latent_df, graph_dict, predictor, do_to_pu, fare_params, fixed_multipliers: np.ndarray, deviator: int) -> list[dict]:
    rows = []
    freeze_iteration = int(args.baseline_eval_episodes)
    unfreeze_iteration = freeze_iteration + int(args.freeze_episodes) if args.unfreeze_at is None else int(args.unfreeze_at)
    for ep in range(args.baseline_eval_episodes):
        env = make_env(args, latent_df, graph_dict, predictor, do_to_pu, fare_params)
        env.reset()
        total_profit = np.zeros(N_FIRMS)
        for _ in range(args.steps):
            _, rewards_by_zone, done, _ = env.step(fixed_multipliers)
            total_profit += rewards_by_zone.sum(axis=1)
            if done:
                break
        if args.log_every and ((ep + 1) % args.log_every == 0 or ep == 0):
            print(f"before freeze eval {ep+1}/{args.baseline_eval_episodes} | avg_mult={np.round(fixed_multipliers.mean(axis=1), 3).tolist()}")
        for firm in range(N_FIRMS):
            rows.append({
                "iteration": ep,
                "phase": "before_freeze_expected_policy_eval",
                "firm_id": firm,
                "profit": float(total_profit[firm]),
                "avg_multiplier": float(fixed_multipliers[firm].mean()),
                "is_deviator": int(firm == deviator),
                "is_frozen": 0,
                "freeze_iteration": freeze_iteration,
                "unfreeze_iteration": unfreeze_iteration,
            })
    return rows


def run_freeze_unfreeze(args, latent_df, graph_dict, predictor, do_to_pu, fare_params, agents: list[ZoneExp3Adapter], fixed_multipliers: np.ndarray, deviator: int) -> list[dict]:
    rows = []
    frozen_firms = [firm for firm in range(N_FIRMS) if firm != deviator]
    freeze_iteration = int(args.baseline_eval_episodes)
    unfreeze_iteration = freeze_iteration + int(args.freeze_episodes) if args.unfreeze_at is None else int(args.unfreeze_at)
    if unfreeze_iteration <= freeze_iteration:
        raise ValueError("Unfreeze iteration must be after freeze iteration.")
    freeze_phase_episodes = unfreeze_iteration - freeze_iteration

    # Phase 1: freeze three platforms; deviator continues learning.
    for ep in range(freeze_phase_episodes):
        env = make_env(args, latent_df, graph_dict, predictor, do_to_pu, fare_params)
        env.reset()
        total_profit = np.zeros(N_FIRMS)
        total_mult = np.zeros(N_FIRMS)
        for _ in range(args.steps):
            multipliers = fixed_multipliers.copy()
            dev_actions = agents[deviator].select_actions()
            multipliers[deviator] = MULTIPLIERS[dev_actions]
            _, rewards_by_zone, done, _ = env.step(multipliers)
            agents[deviator].update(dev_actions, rewards_by_zone[deviator])
            total_profit += rewards_by_zone.sum(axis=1)
            total_mult += multipliers.mean(axis=1)
            if done:
                break
        if args.log_every and ((ep + 1) % args.log_every == 0 or ep == 0):
            print(f"freeze phase {ep+1}/{freeze_phase_episodes} | revenue={np.round(total_profit, 2).tolist()} | avg_mult={np.round(total_mult / args.steps, 3).tolist()}")
        iteration = freeze_iteration + ep
        for firm in range(N_FIRMS):
            rows.append({
                "iteration": iteration,
                "phase": "after_freeze_deviator_learning" if firm == deviator else "after_freeze_expected_policy_frozen",
                "firm_id": firm,
                "profit": float(total_profit[firm]),
                "avg_multiplier": float((total_mult / args.steps)[firm]),
                "is_deviator": int(firm == deviator),
                "is_frozen": int(firm in frozen_firms),
                "freeze_iteration": freeze_iteration,
                "unfreeze_iteration": unfreeze_iteration,
            })

    # Wipe memory from the three frozen agents before unfreezing them
    print("Wiping memory of frozen platforms before unfreezing...")
    for firm in frozen_firms:
        agents[firm].log_weights.fill(0.0)

    # Phase 2: unfreeze all platforms; all four continue learning.
    for ep in range(args.post_unfreeze_episodes):
        env = make_env(args, latent_df, graph_dict, predictor, do_to_pu, fare_params)
        env.reset()
        total_profit = np.zeros(N_FIRMS)
        total_mult = np.zeros(N_FIRMS)
        for _ in range(args.steps):
            actions_by_firm = [agent.select_actions() for agent in agents]
            multipliers = np.vstack([MULTIPLIERS[a] for a in actions_by_firm])
            _, rewards_by_zone, done, _ = env.step(multipliers)
            for firm in range(N_FIRMS):
                agents[firm].update(actions_by_firm[firm], rewards_by_zone[firm])
            total_profit += rewards_by_zone.sum(axis=1)
            total_mult += multipliers.mean(axis=1)
            if done:
                break
        if args.log_every and ((ep + 1) % args.log_every == 0 or ep == 0):
            print(f"post-unfreeze phase {ep+1}/{args.post_unfreeze_episodes} | revenue={np.round(total_profit, 2).tolist()} | avg_mult={np.round(total_mult / args.steps, 3).tolist()}")
        iteration = unfreeze_iteration + ep
        for firm in range(N_FIRMS):
            rows.append({
                "iteration": iteration,
                "phase": "after_unfreeze_all_learning",
                "firm_id": firm,
                "profit": float(total_profit[firm]),
                "avg_multiplier": float((total_mult / args.steps)[firm]),
                "is_deviator": int(firm == deviator),
                "is_frozen": 0,
                "freeze_iteration": freeze_iteration,
                "unfreeze_iteration": unfreeze_iteration,
            })
    return rows


def wide(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = (
        df.groupby(["iteration", "firm_id"], as_index=False)[value_col].mean()
        .pivot(index="iteration", columns="firm_id", values=value_col)
        .fillna(0.0)
        .sort_index()
    )
    for firm in range(N_FIRMS):
        if firm not in out.columns:
            out[firm] = 0.0
    return out[sorted(out.columns)]


def auto_find_monopoly_csv() -> Optional[Path]:
    candidates = [
        Path("tests/logs/monopoly_rewards_history.csv"),
        Path("../tests/logs/monopoly_rewards_history.csv"),
        Path("../../tests/logs/monopoly_rewards_history.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_monopoly_benchmark(args, target_len: int) -> Optional[np.ndarray]:
    path = Path(args.monopoly_csv) if args.monopoly_csv else auto_find_monopoly_csv()
    if path is None:
        print("WARNING: No monopoly benchmark CSV found. Plotting without monopoly line.")
        return None
    if not path.exists():
        raise FileNotFoundError(f"Monopoly CSV not found: {path}")
    df = pd.read_csv(path)
    if args.monopoly_column:
        if args.monopoly_column not in df.columns:
            raise ValueError(f"Column {args.monopoly_column} not found in {path}. Available columns: {list(df.columns)}")
        series = pd.to_numeric(df[args.monopoly_column], errors="coerce")
    else:
        preferred = ["Revenue", "revenue", "total_revenue", "episode_revenue", "Reward", "reward", "Profit", "profit"]
        column = next((col for col in preferred if col in df.columns), None)
        if column is None:
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in ["episode", "iteration", "round", "step"]]
            if not numeric_cols:
                raise ValueError(f"Could not infer monopoly revenue column in {path}. Use --monopoly-column.")
            column = numeric_cols[-1]
        series = pd.to_numeric(df[column], errors="coerce")
    values = series.dropna().to_numpy(dtype=float) * float(args.monopoly_scale)
    if len(values) < target_len:
        values = np.pad(values, (0, target_len - len(values)), mode="edge")
    else:
        values = values[:target_len]
    print(f"Loaded monopoly benchmark from {path} with scale={args.monopoly_scale}")
    return values


def add_phase_lines(ax, freeze_iteration: int, unfreeze_iteration: int) -> None:
    ax.axvline(freeze_iteration, color=FREEZE_COLOR, linestyle="--", linewidth=2.2, label="THE FREEZE")
    ax.axvline(unfreeze_iteration, color=UNFREEZE_COLOR, linestyle=":", linewidth=2.6, label="THE UNFREEZE")
    ymin, ymax = ax.get_ylim()
    ax.text(unfreeze_iteration, ymin + 0.78 * (ymax - ymin), "THE UNFREEZE", rotation=90, ha="left", va="center", color=UNFREEZE_COLOR, fontsize=10)


def plot_revenue(df: pd.DataFrame, out_dir: Path, deviator: int, ma_window: int, dpi: int, monopoly: Optional[np.ndarray], fleet_sizes: list[int]) -> None:
    revenue = wide(df, "revenue_per_taxi")
    freeze = int(df["freeze_iteration"].iloc[0])
    unfreeze = int(df["unfreeze_iteration"].iloc[0])
    x = revenue.index.to_numpy()
    
    total_fleet = sum(fleet_sizes)
    
    # 1. Episode Revenue per Taxi Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    for firm in range(N_FIRMS):
        y = revenue[firm].to_numpy(dtype=float)
        y_ma = moving_average_same_length(y, ma_window)
        is_dev = firm == deviator
        role = "deviator / continuous learning" if is_dev else "frozen, reset, then learning"
        ax.plot(x, y, color=PLATFORM_COLORS[firm], alpha=0.17, linewidth=1.2, linestyle="-" if is_dev else "--")
        ax.plot(x, y_ma, color=PLATFORM_COLORS[firm], linestyle="-" if is_dev else "--", linewidth=3.2 if is_dev else 2.5, label=f"Platform {firm+1} ({role}, {ma_window}-Ep Avg)")
    if monopoly is not None:
        mb = monopoly[:len(x)] / (total_fleet / 4.0)
        ax.plot(x, mb, color="0.25", alpha=0.18, linewidth=1.2, linestyle="-")
        ax.plot(x, moving_average_same_length(mb, ma_window), color=MONOPOLY_COLOR, linestyle="-", linewidth=3.4, label=f"Monopoly ({ma_window}-Ep Avg)")
    add_phase_lines(ax, freeze, unfreeze)
    ax.set_title(f"Courthoud Defection Test Revenue per Taxi Comparison — Platform {deviator+1} Deviates (Reset Frozen)", fontsize=18, weight="bold")
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Revenue per Taxi ($)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"freeze_episode_total_revenue_with_monopoly_firm_{deviator}_deviates.png", dpi=dpi)
    plt.close(fig)

    # 2. Cumulative Revenue per Taxi Plot
    cumulative = revenue.cumsum()
    fig, ax = plt.subplots(figsize=(14, 7))
    for firm in range(N_FIRMS):
        y = cumulative[firm].to_numpy(dtype=float)
        is_dev = firm == deviator
        role = "deviator / continuous learning" if is_dev else "frozen, reset, then learning"
        ax.plot(x, y, color=PLATFORM_COLORS[firm], linestyle="-" if is_dev else "--", linewidth=3.2 if is_dev else 2.5, label=f"Platform {firm+1} ({role})")
    if monopoly is not None:
        mb_cum = np.cumsum(monopoly[:len(x)]) / (total_fleet / 4.0)
        ax.plot(x, mb_cum, color=MONOPOLY_COLOR, linestyle="-", linewidth=3.4, label="Monopoly")
    add_phase_lines(ax, freeze, unfreeze)
    ax.set_title(f"Courthoud Defection Test Cumulative Revenue per Taxi — Platform {deviator+1} Deviates (Reset Frozen)", fontsize=18, weight="bold")
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Cumulative Revenue per Taxi ($)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"freeze_cumulative_total_revenue_with_monopoly_firm_{deviator}_deviates.png", dpi=dpi)
    plt.close(fig)

def save_summary(df: pd.DataFrame, out_dir: Path, deviator: int, fleet_sizes: list[int]) -> None:
    revenue = wide(df, "profit")
    freeze = int(df["freeze_iteration"].iloc[0])
    unfreeze = int(df["unfreeze_iteration"].iloc[0])
    rows = []
    for firm in range(N_FIRMS):
        before = revenue.index < freeze
        freeze_phase = (revenue.index >= freeze) & (revenue.index < unfreeze)
        after = revenue.index >= unfreeze
        
        fleet_size = fleet_sizes[firm]
        
        rows.append({
            "deviator_firm_id": deviator,
            "firm_id": firm,
            "role": "deviator" if firm == deviator else "frozen_reset_learning",
            "fleet_size": fleet_size,
            "before_freeze_mean_revenue": float(revenue.loc[before, firm].mean()) if before.any() else np.nan,
            "freeze_phase_mean_revenue": float(revenue.loc[freeze_phase, firm].mean()) if freeze_phase.any() else np.nan,
            "post_unfreeze_mean_revenue": float(revenue.loc[after, firm].mean()) if after.any() else np.nan,
            "final_cumulative_revenue": float(revenue[firm].cumsum().iloc[-1]),
            "before_freeze_mean_rev_per_taxi": float(revenue.loc[before, firm].mean() / fleet_size) if before.any() else np.nan,
            "freeze_phase_mean_rev_per_taxi": float(revenue.loc[freeze_phase, firm].mean() / fleet_size) if freeze_phase.any() else np.nan,
            "post_unfreeze_mean_rev_per_taxi": float(revenue.loc[after, firm].mean() / fleet_size) if after.any() else np.nan,
            "final_cumulative_rev_per_taxi": float(revenue[firm].cumsum().iloc[-1] / fleet_size),
        })
    pd.DataFrame(rows).to_csv(out_dir / f"freeze_summary_with_monopoly_firm_{deviator}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-file Courthoud freeze/unfreeze experiment with 1/4 monopoly benchmark plot (Reset Frozen)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="experiments/courthoud_freeze_unfreeze_monopoly_reset")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--baseline-eval-episodes", type=int, default=70)
    parser.add_argument("--freeze-episodes", type=int, default=70)
    parser.add_argument("--unfreeze-at", type=int, default=None)
    parser.add_argument("--post-unfreeze-episodes", type=int, default=70)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--taxis", type=int, nargs="+", default=[5000])
    parser.add_argument("--n-states", type=int, default=10)
    parser.add_argument("--bin-size", type=int, default=10)
    parser.add_argument("--theta", type=float, default=0.4)
    parser.add_argument("--start-day", type=int, default=0)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--reward-normalizer", type=float, default=50000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-deviator", action="store_true")
    parser.add_argument("--deviator", type=int, default=0)
    parser.add_argument("--ma-window", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--monopoly-csv", type=str, default=None)
    parser.add_argument("--monopoly-column", type=str, default=None)
    parser.add_argument("--monopoly-scale", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_df, graph_dict, predictor, do_to_pu, fare_params = load_inputs(Path(args.data_dir))
    baseline_agents = train_baseline(args, latent_df, graph_dict, predictor, do_to_pu, fare_params)

    rng = np.random.default_rng(args.seed)
    deviator = int(rng.integers(0, N_FIRMS)) if args.random_deviator else int(args.deviator)
    frozen_firms = [firm for firm in range(N_FIRMS) if firm != deviator]
    print(f"deviator = Platform {deviator + 1}; frozen = {[firm + 1 for firm in frozen_firms]}")

    agents = copy.deepcopy(baseline_agents)
    fixed_multipliers = agents_to_expected_multipliers(agents)
    print("expected-policy frozen avg multipliers:", np.round(fixed_multipliers.mean(axis=1), 3))

    before_rows = evaluate_before_freeze(args, latent_df, graph_dict, predictor, do_to_pu, fare_params, fixed_multipliers, deviator)
    after_rows = run_freeze_unfreeze(args, latent_df, graph_dict, predictor, do_to_pu, fare_params, agents, fixed_multipliers, deviator)
    df = pd.DataFrame(before_rows + after_rows)
    fleet_sizes = [args.taxis[0] // 4] * 4 if len(args.taxis) == 1 else [int(x) for x in args.taxis]
    df["revenue_per_taxi"] = df.apply(lambda row: row["profit"] / fleet_sizes[int(row["firm_id"])], axis=1)

    timeline_path = out_dir / f"courthoud_freeze_unfreeze_timeline_deviator_firm_{deviator}.csv"
    df.to_csv(timeline_path, index=False)

    target_len = int(df["iteration"].max()) + 1
    monopoly = load_monopoly_benchmark(args, target_len)
    plot_revenue(df, out_dir, deviator, args.ma_window, args.dpi, monopoly, fleet_sizes)
    save_summary(df, out_dir, deviator, fleet_sizes)

    print(f"saved timeline: {timeline_path}")
    print(f"saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
