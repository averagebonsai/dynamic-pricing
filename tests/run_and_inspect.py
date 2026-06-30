import sys
from pathlib import Path
import numpy as np
import polars as pl
import pickle
from sklearn.linear_model import LinearRegression

# Add project src to path dynamically
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from experiment_pipeline import MonopolyTaxiEnv, OligopolyTaxiEnv
from destination import DestinationNNPredictor
from iql import run_iql
from rl_algorithms import Exp3Agent

# Setup parameters
class Args:
    taxis = 5000
    n_states = 1
    bin_size = 10
    theta = 0.4
    steps = 24
    episodes = 100
    alpha = 0.2
    discount = 0.95
    eps = 0.1
    gamma = 0.1
    eta = 0.1

args = Args()

# Load data
data_dir = PROJECT_DIR / "data"
latent_df = pl.read_csv(data_dir / "latent_mle_params.csv")
graph_df = pl.read_csv(data_dir / "graph.csv")

with open(data_dir / "destination_predictor.pkl", "rb") as f:
    predictor = pickle.load(f)

do_to_pu = {}
for do_val in predictor.unique_do:
    if do_val in predictor.unique_pu:
        do_to_pu[do_val] = do_val
    else:
        do_to_pu[do_val] = predictor.unique_pu[0]

graph_dict = {}
for row in graph_df.iter_rows(named=True):
    pu = row["PULocationID"]
    do = row["DOLocationID"]
    if pu not in graph_dict:
        graph_dict[pu] = {}
    graph_dict[pu][do] = {
        "fare": row["baseline_fare"]
    }

x_reg = graph_df["baseline_distance"].to_numpy().reshape(-1, 1)
y_reg = graph_df["baseline_fare"].to_numpy()
reg_model = LinearRegression().fit(x_reg, y_reg)
default_fare_params = {
    "price_per_mile": float(reg_model.coef_[0]),
    "intercept": float(reg_model.intercept_)
}

# Run Monopoly
print("--- Running Monopoly ---")
env_mono = MonopolyTaxiEnv(
    latent_df=latent_df,
    graph_dict=graph_dict,
    predictor=predictor,
    do_to_pu=do_to_pu,
    fleet_size=args.taxis,
    n_states=args.n_states,
    bin_size=args.bin_size,
    theta=args.theta,
    steps_per_episode=args.steps,
    price_per_mile=default_fare_params["price_per_mile"],
    intercept=default_fare_params["intercept"]
)

q_tables, rewards_history_mono = run_iql(
    env=env_mono,
    n_agents=env_mono.n_arms,
    n_states=args.n_states,
    n_actions=5,
    episodes=args.episodes,
    steps_per_episode=args.steps,
    alpha=args.alpha,
    discount=args.discount,
    eps=args.eps
)

# Run one final eval episode to check actions for Monopoly
states = env_mono.reset()
mono_mults = []
for step in range(args.steps):
    actions = np.zeros(env_mono.n_arms, dtype=int)
    for i in range(env_mono.n_arms):
        actions[i] = np.argmax(q_tables[i, states[i]])
    multipliers_options = [1.0, 1.2, 1.5, 1.8, 2.0]
    mono_mults.extend([multipliers_options[a] for a in actions])
    states, _, _, _ = env_mono.step(actions)

print(f"Monopoly Average Multiplier: {np.mean(mono_mults):.4f}")

# Run Oligopoly
print("--- Running Oligopoly ---")
env_oligo = OligopolyTaxiEnv(
    latent_df=latent_df,
    graph_dict=graph_dict,
    predictor=predictor,
    do_to_pu=do_to_pu,
    fleet_size=args.taxis,
    n_states=args.n_states,
    bin_size=args.bin_size,
    theta=args.theta,
    steps_per_episode=args.steps,
    price_per_mile=default_fare_params["price_per_mile"],
    intercept=default_fare_params["intercept"]
)

multipliers = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
agents = [
    Exp3Agent(n_arms=env_oligo.n_arms, n_actions=4, gamma=args.gamma, eta=args.eta)
    for _ in range(4)
]

for ep in range(args.episodes):
    states = env_oligo.reset()
    for step in range(args.steps):
        actions_indices = []
        actions_mults = []
        for k in range(4):
            acts = agents[k].select_actions()
            actions_indices.append(acts)
            actions_mults.append(multipliers[acts])

        next_states, rewards, done, _ = env_oligo.step(np.array(actions_mults))

        for k in range(4):
            taxis = env_oligo.taxis[k]
            scaled_rewards = np.zeros(env_oligo.n_arms)
            for arm in range(env_oligo.n_arms):
                n_taxis = taxis[arm]
                if n_taxis > 0:
                    avg_fare = rewards[k, arm] / n_taxis
                    scaled_rewards[arm] = np.clip(avg_fare / 270.0, 0.0, 1.0)
                else:
                    scaled_rewards[arm] = 0.0
            agents[k].update(actions_indices[k], scaled_rewards)
        states = next_states

# Run one final eval episode to check actions for Oligopoly
states = env_oligo.reset()
oligo_mults = []
for step in range(args.steps):
    actions_mults = []
    for k in range(4):
        # Greedy choice from weights
        acts = np.zeros(env_oligo.n_arms, dtype=int)
        for arm in range(env_oligo.n_arms):
            acts[arm] = np.argmax(agents[k].log_weights[arm])
        actions_mults.append(multipliers[acts])
    oligo_mults.extend(np.array(actions_mults).flatten())
    env_oligo.step(np.array(actions_mults))

print(f"Oligopoly Average Multiplier: {np.mean(oligo_mults):.4f}")
