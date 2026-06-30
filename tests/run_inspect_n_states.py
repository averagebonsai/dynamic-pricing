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

from experiment_pipeline import MonopolyTaxiEnv
from destination import DestinationNNPredictor
from iql import run_iql

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

for n_states in [1, 10]:
    np.random.seed(42)
    env = MonopolyTaxiEnv(
        latent_df=latent_df,
        graph_dict=graph_dict,
        predictor=predictor,
        do_to_pu=do_to_pu,
        fleet_size=5000,
        n_states=n_states,
        bin_size=10,
        theta=0.4,
        steps_per_episode=24,
        price_per_mile=default_fare_params["price_per_mile"],
        intercept=default_fare_params["intercept"]
    )

    q_tables, rewards_history = run_iql(
        env=env,
        n_agents=env.n_arms,
        n_states=n_states,
        n_actions=5,
        episodes=100,
        steps_per_episode=24,
        alpha=0.2,
        discount=0.95,
        eps=0.1
    )

    # Eval episode
    states = env.reset()
    mono_mults = []
    for step in range(24):
        actions = np.zeros(env.n_arms, dtype=int)
        for i in range(env.n_arms):
            actions[i] = np.argmax(q_tables[i, states[i]])
        multipliers_options = [1.0, 1.2, 1.5, 1.8, 2.0]
        mono_mults.extend([multipliers_options[a] for a in actions])
        states, _, _, _ = env.step(actions)

    print(f"n_states={n_states} | Final Episode Revenue: ${rewards_history[-1]:,.2f} | Average Multiplier: {np.mean(mono_mults):.4f}")
