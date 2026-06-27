import sys
from pathlib import Path
import numpy as np
import polars as pl
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Add project src to path dynamically
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from experiment_pipeline import MonopolyTaxiEnv, OligopolyTaxiEnv
from destination import DestinationNNPredictor
from rl_algorithms import Exp3Agent


### Monopoly run loop with aggregated interval multiplier logging and revenue plotting.
### Collects average surge multipliers chosen per zone across 10-episode intervals, 
### saves the final consolidated table to a summary CSV, and plots the total episode revenue.

def run_monopoly_with_logging(args, env, default_fare_params):
    print("\n--- Running Monopoly (IQL) with Aggregated Multiplier Logging ---")
    
    n_arms = env.n_arms
    q_tables = np.zeros((n_arms, args.n_states, 5), dtype=float)
    multipliers_options = [1.0, 1.2, 1.5, 1.8, 2.0]
    
    # Track actions taken
    ep_multiplier_history = []
    rewards_history = []
    
    # Store aggregated averages: shape (n_arms, intervals)
    intervals = args.episodes // 10
    summary_data = np.zeros((n_arms, intervals))
    
    for ep in range(args.episodes):
        states = np.asarray(env.reset(), dtype=int)
        step_mults = []
        total_reward = 0.0
        
        for step in range(args.steps):
            actions = np.zeros(n_arms, dtype=int)
            for i in range(n_arms):
                if np.random.rand() < args.eps:
                    actions[i] = np.random.randint(5)
                else:
                    actions[i] = np.argmax(q_tables[i, states[i]])
            
            # Record multipliers
            step_mults.append([multipliers_options[a] for a in actions])
            
            next_states, rewards, done, _ = env.step(actions)
            next_states = np.asarray(next_states, dtype=int)
            rewards = np.asarray(rewards, dtype=float)
            total_reward += float(np.sum(rewards))
            
            # IQL Update
            for i in range(n_arms):
                s = states[i]
                a = actions[i]
                ns = next_states[i]
                r = rewards[i]
                
                n_taxis = env.taxis[i]
                if n_taxis > 0:
                    avg_fare = r / n_taxis
                    scaled_r = np.clip(avg_fare / 100.0, 0.0, 1.0)
                else:
                    scaled_r = 0.0
                
                target = scaled_r + args.discount * np.max(q_tables[i, ns])
                q_tables[i, s, a] += args.alpha * (target - q_tables[i, s, a])
                
            states = next_states
            if done:
                break
                
        ep_multiplier_history.append(step_mults)
        rewards_history.append(total_reward)
        
        # Every 10 episodes, aggregate average multipliers
        if (ep + 1) % 10 == 0:
            interval_idx = (ep + 1) // 10 - 1
            last_10 = ep_multiplier_history[-10:]
            avg_mults = np.mean(last_10, axis=(0, 1))
            summary_data[:, interval_idx] = avg_mults
            
            print(f"[Episode {ep + 1}/{args.episodes}] Monopoly Avg Multiplier: {np.mean(avg_mults):.4f} | Revenue: ${total_reward:,.2f}")
            
    # Save unified summary CSV
    log_dir = TEST_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    csv_path = log_dir / "monopoly_multipliers_summary.csv"
    
    df_dict = {
        "Zone_Index": list(range(n_arms)),
        "PULocationID": [env.idx_to_pu[i] for i in range(n_arms)]
    }
    for i in range(intervals):
        df_dict[f"{i*10}-{(i+1)*10}"] = list(np.round(summary_data[:, i], 2))
        
    df = pl.DataFrame(df_dict)
    df.write_csv(csv_path)
    print(f"\nMonopoly experiment finished. Aggregated summary saved to: [monopoly_multipliers_summary.csv](file://{csv_path})")
    
    # Plot Monopoly results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label="Episode Revenue", color="#1f77b4", alpha=0.4)
    window = min(10, len(rewards_history))
    moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(rewards_history)), moving_avg, label=f"{window}-Ep Moving Avg", color="#005b96", linewidth=2.5)
    
    plt.title("Monopoly Ride-Hailing Independent Q-Learning (IQL) Revenue Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Revenue ($)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    out_file = TEST_DIR / "monopoly_results.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Revenue plot saved to: [monopoly_results.png](file://{out_file})")


### Oligopoly run loop with aggregated interval multiplier logging and revenue plotting.
### Collects average surge multipliers per zone per platform across 10-episode intervals,
### outputs consolidated platform summary CSVs, and plots the platform revenue comparison.

def run_oligopoly_with_logging(args, env, default_fare_params):
    print("\n--- Running Oligopoly (EXP3) with Aggregated Multiplier Logging ---")
    
    n_arms = env.n_arms
    n_agents = 4
    multipliers = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
    agents = [
        Exp3Agent(n_arms=n_arms, n_actions=4, gamma=args.gamma, eta=args.eta)
        for _ in range(n_agents)
    ]
    
    # Track actions and rewards
    ep_multiplier_history = []
    agent_rewards_history = [[] for _ in range(n_agents)]
    
    # Store aggregated averages: shape (4, n_arms, intervals)
    intervals = args.episodes // 10
    summary_data = np.zeros((4, n_arms, intervals))
    
    for ep in range(args.episodes):
        states = env.reset()
        step_mults = []
        ep_rewards = np.zeros(n_agents)
        
        for step in range(args.steps):
            actions_indices = []
            actions_mults = []
            for k in range(n_agents):
                acts = agents[k].select_actions()
                actions_indices.append(acts)
                actions_mults.append(multipliers[acts])
                
            # Record multipliers
            step_mults.append(actions_mults)
            
            next_states, rewards, done, _ = env.step(np.array(actions_mults))
            
            # Accumulate step rewards and update agents
            for k in range(n_agents):
                ep_rewards[k] += float(np.sum(rewards[k]))
                taxis = env.taxis[k]
                scaled_rewards = np.zeros(n_arms)
                for arm in range(n_arms):
                    n_taxis = taxis[arm]
                    if n_taxis > 0:
                        avg_fare = rewards[k, arm] / n_taxis
                        scaled_rewards[arm] = np.clip(avg_fare / 270.0, 0.0, 1.0)
                    else:
                        scaled_rewards[arm] = 0.0
                agents[k].update(actions_indices[k], scaled_rewards)
                
            states = next_states
            if done:
                break
                
        ep_multiplier_history.append(step_mults)
        for k in range(n_agents):
            agent_rewards_history[k].append(ep_rewards[k])
        
        # Every 10 episodes, aggregate average multipliers
        if (ep + 1) % 10 == 0:
            interval_idx = (ep + 1) // 10 - 1
            last_10 = ep_multiplier_history[-10:]
            avg_mults = np.mean(last_10, axis=(0, 1)) # shape (4, n_arms)
            summary_data[:, :, interval_idx] = avg_mults
            
            print(f"[Episode {ep + 1}/{args.episodes}] Oligopoly Avg Multiplier (All Platforms): {np.mean(avg_mults):.4f} | Total Rev: ${np.sum(ep_rewards):,.2f}")
            
    # Save one summary CSV per platform
    log_dir = TEST_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    print("\nOligopoly experiment finished. Platform summary CSVs saved to:")
    for k in range(n_agents):
        csv_path = log_dir / f"oligopoly_platform_{k+1}_summary.csv"
        df_dict = {
            "Zone_Index": list(range(n_arms)),
            "PULocationID": [env.idx_to_pu[i] for i in range(n_arms)]
        }
        for i in range(intervals):
            df_dict[f"{i*10}-{(i+1)*10}"] = list(np.round(summary_data[k, :, i], 2))
            
        df = pl.DataFrame(df_dict)
        df.write_csv(csv_path)
        print(f"  Platform {k+1}: [oligopoly_platform_{k+1}_summary.csv](file://{csv_path})")
        
    # Plot Oligopoly results
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k in range(n_agents):
        plt.plot(agent_rewards_history[k], alpha=0.3, color=colors[k])
        window = min(10, len(agent_rewards_history[k]))
        moving_avg = np.convolve(agent_rewards_history[k], np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(agent_rewards_history[k])), moving_avg, 
                 label=f"Platform {k+1} ({window}-Ep Avg)", color=colors[k], linewidth=2)
                 
    plt.title("Oligopoly Competition (4-Agent EXP3) Revenue Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Revenue ($)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    out_file = TEST_DIR / "oligopoly_results.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Revenue comparison plot saved to: [oligopoly_results.png](file://{out_file})")


### Main entry point parsing arguments, loading graph & demand data, and launching the specified simulation.

def main():
    parser = argparse.ArgumentParser(description="Run Taxi pricing simulations with multiplier logging")
    parser.add_argument("--mode", type=str, choices=["monopoly", "oligopoly"], default="monopoly",
                        help="Choose monopoly or oligopoly simulation")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=120, help="Steps per episode")
    parser.add_argument("--taxis", type=int, default=5000, help="Fleet size")
    parser.add_argument("--n-states", type=int, default=1, help="Number of states")
    parser.add_argument("--bin-size", type=int, default=10, help="Discretization bin size")
    parser.add_argument("--theta", type=float, default=0.4, help="Price sensitivity theta")
    parser.add_argument("--discount", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    
    # monopoly args
    parser.add_argument("--alpha", type=float, default=0.2, help="IQL alpha")
    parser.add_argument("--eps", type=float, default=0.1, help="IQL epsilon")
    
    # oligopoly args
    parser.add_argument("--gamma", type=float, default=0.1, help="EXP3 gamma")
    parser.add_argument("--eta", type=float, default=0.1, help="EXP3 eta")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    data_dir = PROJECT_DIR / "data"
    print(f"Loading data from {data_dir}...")
    latent_df = pl.read_csv(data_dir / "latent_mle_params.csv")
    graph_df = pl.read_csv(data_dir / "graph.csv")
    
    predictor_path = data_dir / "destination_predictor.pkl"
    with open(predictor_path, "rb") as f:
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
    
    if args.mode == "monopoly":
        env = MonopolyTaxiEnv(
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
        run_monopoly_with_logging(args, env, default_fare_params)
    elif args.mode == "oligopoly":
        env = OligopolyTaxiEnv(
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
        run_oligopoly_with_logging(args, env, default_fare_params)

if __name__ == "__main__":
    main()
