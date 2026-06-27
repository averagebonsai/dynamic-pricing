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

# Customized run loop for fast testing
def run_fast_iql(alpha, normalizer, n_states=1):
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
    
    q_tables = np.zeros((env.n_arms, n_states, 5), dtype=float)
    rewards_history = []
    
    for ep in range(100):
        states = np.asarray(env.reset(), dtype=int)
        total_reward = 0.0
        
        for step in range(24):
            actions = np.zeros(env.n_arms, dtype=int)
            for i in range(env.n_arms):
                if np.random.rand() < 0.1:
                    actions[i] = np.random.randint(5)
                else:
                    actions[i] = np.argmax(q_tables[i, states[i]])
                    
            next_states, rewards, done, _ = env.step(actions)
            next_states = np.asarray(next_states, dtype=int)
            rewards = np.asarray(rewards, dtype=float)
            
            for i in range(env.n_arms):
                s = states[i]
                a = actions[i]
                ns = next_states[i]
                r = rewards[i]
                
                n_taxis = env.taxis[i]
                if n_taxis > 0:
                    avg_fare = r / n_taxis
                    scaled_r = np.clip(avg_fare / normalizer, 0.0, 1.0)
                else:
                    scaled_r = 0.0

                target = scaled_r + 0.95 * np.max(q_tables[i, ns])
                q_tables[i, s, a] += alpha * (target - q_tables[i, s, a])
                
            total_reward += float(np.sum(rewards))
            states = next_states
            if done:
                break
        rewards_history.append(total_reward)
        
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
        
    return rewards_history[-1], np.mean(mono_mults)

for norm in [50.0, 100.0, 270.0]:
    for a in [0.05, 0.1, 0.2, 0.3]:
        rev, mult = run_fast_iql(a, norm)
        print(f"Norm={norm} | Alpha={a} | Rev: ${rev:,.2f} | Avg Mult: {mult:.4f}")
