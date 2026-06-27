import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Add src and project root to Python search path for robust imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import algorithms and predictor
from rmab_algorithms import rmab_q_learning, unpack_step, rmab_index_policy
from rl_algorithms import Exp3Agent
from destination import DestinationNNPredictor
from iql import run_iql


class MonopolyTaxiEnv:
    """
    Environment wrapper for single-agent IQL simulation.
    Each taxi pickup zone is modeled as an arm in the bandit.
    The state of each arm is the discretized number of taxis currently in the zone.
    (currently stateless) <-- in future, to implement 3 states. [low supply, meet supply, oversupply]. 
    """
    def __init__(
        self,
        latent_df: pl.DataFrame, #latent taxi demand
        graph_dict: dict, #graph with baseline distances, times
        predictor: DestinationNNPredictor, #probabilities from origin --> destination
        do_to_pu: dict, #drop-off to pick-up mapping
        fleet_size: int = 5000,
        n_states: int = 10, #number of bins --> how many "states of taxis" are there. 
        bin_size: int = 10, #how many in each bin --> the idea is that the status of each zone is not an integer num of taxis, but a range (e.g [0, 9])
        theta: float = 0.4,
        steps_per_episode: int = 120,
        start_day: int = 0,
        start_hour: int = 0,
        default_distance: float = 5.0,
        price_per_mile: float = 3.0,
        intercept: float = 8.0,
        active_multiplier: float = 1.5
    ):
        self.latent_df = latent_df
        self.graph_dict = graph_dict
        self.predictor = predictor
        self.do_to_pu = do_to_pu
        self.fleet_size = fleet_size
        self.n_states = n_states
        self.bin_size = bin_size
        self.theta = theta
        self.steps_per_episode = steps_per_episode
        self.start_day = start_day
        self.start_hour = start_hour
        self.default_distance = default_distance
        self.price_per_mile = price_per_mile
        self.intercept = intercept
        self.active_multiplier = active_multiplier # <-- to modify to change to a few strategies, [1.0, 1.2, 1.5, 1.8, 2.0]

        self.n_arms = len(self.predictor.unique_pu) #number of zones = 262
        self.pu_to_idx = {pu: idx for idx, pu in enumerate(self.predictor.unique_pu)}
        self.idx_to_pu = {idx: pu for idx, pu in enumerate(self.predictor.unique_pu)}

        # Build demand lookup dictionary: (pu_id, day, hour) -> latent_poisson
        self.demand_lookup = {}
        for row in latent_df.iter_rows(named=True):
            pu = row["PULocationID"]
            day = row["day_of_week"]
            hour = row["request_hour"]
            val = row["latent_poisson"]
            if val is None or np.isnan(val):
                val = row["historical_poisson"]
                if val is None or np.isnan(val):
                    val = 0.0
            self.demand_lookup[(pu, day, hour)] = val

        self.taxis = np.zeros(self.n_arms, dtype=int) # <-- to modify 
        self.current_step = 0
        self.prediction_cache = {}

    def reset(self) -> np.ndarray:
        self.current_step = 0

        ## ---> KIV: To amend once the taxi start-state code is set-up <---
        # This current code: Distribute taxis proportional to average latent demand across the week
        # Taxis appear to meet demand, not realistic. 
        
        pu_avg_demand = np.zeros(self.n_arms)
        for idx, pu in self.idx_to_pu.items():
            demands = [self.demand_lookup.get((pu, d, h), 0.0) for d in range(7) for h in range(24)]
            pu_avg_demand[idx] = np.mean(demands)

        if pu_avg_demand.sum() > 0:
            probs = pu_avg_demand / pu_avg_demand.sum()
        else:
            probs = np.ones(self.n_arms) / self.n_arms

        # Draw initial taxi counts
        self.taxis = np.random.multinomial(self.fleet_size, probs)
        return self._get_states()

    def _get_states(self) -> np.ndarray:
        # Discretize taxi counts -- reduce number of states / strategies / arms, makes problem simpler to solve (small state space)
        states = np.minimum(self.taxis // self.bin_size, self.n_states - 1)
        return states.astype(int)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        # Compute current time
        total_hours = self.start_hour + self.current_step
        hour = total_hours % 24
        day = (self.start_day + total_hours // 24) % 7

        rewards = np.zeros(self.n_arms)
        next_taxis = np.zeros(self.n_arms, dtype=int) #array to get the number of taxis for the start of next round??

        for idx in range(self.n_arms): #iterate over every zone. note that index is sequential, PU zones are not, hence the mapping.
            pu = self.idx_to_pu[idx]
            taxis_here = self.taxis[idx]
            action = actions[idx] #actions is a 262-length array where every entry is {0, 1}, in each zone activate strategy or not. currently Boolean. 
            ## needs to be modified, the agent should choose a number instead of a boolean strategy of "activate or not"

            # Multiplier choice: action index corresponds to options [1.0, 1.2, 1.5, 1.8, 2.0]
            multipliers_options = [1.0, 1.2, 1.5, 1.8, 2.0]
            multiplier = multipliers_options[action]

            # Latent demand rate
            lambda_latent = self.demand_lookup.get((pu, day, hour), 0.0)

            # Customer acceptance rate = exp(-theta * (surge - 1))
            lambda_actual = lambda_latent * np.exp(-self.theta * (multiplier - 1.0)) #generates observed demand 
            demand = np.random.poisson(lambda_actual) ### <--- the actual number of people that end up ordering a taxi

            # Matching step
            matched = min(demand, taxis_here)
            unmatched = taxis_here - matched

            # Unmatched taxis stay in the zone
            next_taxis[idx] += unmatched

            if matched > 0: #note: still on a single zone here. 
                # Sample destinations
                cache_key = (pu, day, hour)
                if cache_key not in self.prediction_cache: #prevents recalculation -- if it's in cache, then just calls results from there. 
                    nn_probs = self.predictor.predict(pu, day, hour) #passenger distribution. 
                    sim_probs = np.zeros(self.n_arms) 
                    for do_idx, do_val in self.predictor.idx_to_do.items(): #each drop-off location has a index. 
                        target_pu = self.do_to_pu.get(do_val) # get target pick-up point (for drop-off)
                        if target_pu in self.pu_to_idx: 
                            #simulation probability of index corresponding to pickup += neural network probs of drop-off index. 
                            sim_probs[self.pu_to_idx[target_pu]] += nn_probs[do_idx] 
                    if sim_probs.sum() > 0:
                        sim_probs /= sim_probs.sum()
                    else:
                        sim_probs = np.ones(self.n_arms) / self.n_arms #equal probability across all states
                    self.prediction_cache[cache_key] = sim_probs
                else:
                    sim_probs = self.prediction_cache[cache_key]

                destinations = np.random.choice(self.n_arms, size=matched, p=sim_probs) #why p=sim_probs here instead of p=nn_probs? 

                # Move matched taxis and accumulate fare rewards
                for dest in destinations: #for every taxi
                    next_taxis[dest] += 1 
                    dest_pu = self.idx_to_pu[dest]
                    fare_info = self.graph_dict.get(pu, {}).get(dest_pu) 
                    if fare_info:
                        base_fare = fare_info["fare"] 
                    else:
                        # Regression fallback
                        base_fare = self.default_distance * self.price_per_mile + self.intercept

                    fare = base_fare * multiplier
                    rewards[idx] += fare

        self.taxis = next_taxis
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        return self._get_states(), rewards, done, {}


class OligopolyTaxiEnv:
    """
    Environment wrapper for multi-agent Oligopoly simulation with 4 taxi platforms competing.
    """
    def __init__(
        self,
        latent_df: pl.DataFrame,
        graph_dict: dict,
        predictor: DestinationNNPredictor,
        do_to_pu: dict,
        fleet_size: int = 5000,
        n_states: int = 10,
        bin_size: int = 10,
        theta: float = 0.4,
        steps_per_episode: int = 120,
        start_day: int = 0,
        start_hour: int = 0,
        default_distance: float = 5.0,
        price_per_mile: float = 3.0,
        intercept: float = 8.0
    ):
        self.latent_df = latent_df
        self.graph_dict = graph_dict
        self.predictor = predictor
        self.do_to_pu = do_to_pu
        self.fleet_size = fleet_size
        self.n_states = n_states
        self.bin_size = bin_size
        self.theta = theta
        self.steps_per_episode = steps_per_episode
        self.start_day = start_day
        self.start_hour = start_hour
        self.default_distance = default_distance
        self.price_per_mile = price_per_mile
        self.intercept = intercept

        self.n_arms = len(self.predictor.unique_pu)
        self.pu_to_idx = {pu: idx for idx, pu in enumerate(self.predictor.unique_pu)}
        self.idx_to_pu = {idx: pu for idx, pu in enumerate(self.predictor.unique_pu)}

        # Build demand lookup dictionary: (pu_id, day, hour) -> latent_poisson
        self.demand_lookup = {}
        for row in latent_df.iter_rows(named=True):
            pu = row["PULocationID"]
            day = row["day_of_week"]
            hour = row["request_hour"]
            val = row["latent_poisson"]
            if val is None or np.isnan(val):
                val = row["historical_poisson"]
                if val is None or np.isnan(val):
                    val = 0.0
            self.demand_lookup[(pu, day, hour)] = val

        self.taxis = np.zeros((4, self.n_arms), dtype=int)
        self.current_step = 0
        self.prediction_cache = {}

    def reset(self) -> np.ndarray:
        self.current_step = 0

        # Distribute taxis based on average historical demand
        pu_avg_demand = np.zeros(self.n_arms)
        for idx, pu in self.idx_to_pu.items():
            demands = [self.demand_lookup.get((pu, d, h), 0.0) for d in range(7) for h in range(24)]
            pu_avg_demand[idx] = np.mean(demands)

        if pu_avg_demand.sum() > 0:
            probs = pu_avg_demand / pu_avg_demand.sum()
        else:
            probs = np.ones(self.n_arms) / self.n_arms

        # Equal share of fleet for each platform
        platform_fleet = self.fleet_size // 4
        for k in range(4):
            self.taxis[k] = np.random.multinomial(platform_fleet, probs)

        return self._get_states()

    def _get_states(self) -> np.ndarray:
        # Discretize taxi counts for each platform in each zone
        states = np.minimum(self.taxis // self.bin_size, self.n_states - 1)
        return states.astype(int)

    def step(self, agent_multipliers: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        # agent_multipliers has shape (4, n_arms)
        total_hours = self.start_hour + self.current_step
        hour = total_hours % 24
        day = (self.start_day + total_hours // 24) % 7

        rewards = np.zeros((4, self.n_arms))
        next_taxis = np.zeros((4, self.n_arms), dtype=int)

        for idx in range(self.n_arms):
            pu = self.idx_to_pu[idx]
            mults = agent_multipliers[:, idx] # multipliers of 4 platforms: shape (4,)

            # Sample latent demand
            lambda_latent = self.demand_lookup.get((pu, day, hour), 0.0)
            D_latent = np.random.poisson(lambda_latent)

            if D_latent > 0:
                ### Calculate choice probabilities using a normalized demand model.
                ### The overall market acceptance probability matches a single-operator demand curve at the average price,
                ### preventing artificial market-size inflation from multi-platform competition. Accepted passengers
                ### are distributed among competing platforms proportional to their pricing utility.

                v_vals = np.exp(-self.theta * (mults - 1.0))
                sum_v = np.sum(v_vals)
                avg_mult = np.mean(mults)
                p_accept = np.exp(-self.theta * (avg_mult - 1.0))
                
                probs = np.zeros(5)
                if sum_v > 0:
                    probs[:4] = p_accept * (v_vals / sum_v)
                else:
                    probs[:4] = p_accept * 0.25
                probs[4] = 1.0 - p_accept

                # Sample customer allocation
                demands = np.random.multinomial(D_latent, probs)
            else:
                demands = np.zeros(5, dtype=int)

            # Get cached destination distribution
            cache_key = (pu, day, hour)
            if cache_key not in self.prediction_cache:
                nn_probs = self.predictor.predict(pu, day, hour)
                sim_probs = np.zeros(self.n_arms)
                for do_idx, do_val in self.predictor.idx_to_do.items():
                    target_pu = self.do_to_pu.get(do_val)
                    if target_pu in self.pu_to_idx:
                        sim_probs[self.pu_to_idx[target_pu]] += nn_probs[do_idx]
                if sim_probs.sum() > 0:
                    sim_probs /= sim_probs.sum()
                else:
                    sim_probs = np.ones(self.n_arms) / self.n_arms
                self.prediction_cache[cache_key] = sim_probs
            else:
                sim_probs = self.prediction_cache[cache_key]

            # Matching and relocation for each platform
            for k in range(4):
                taxis_here = self.taxis[k, idx]
                platform_demand = demands[k]

                matched = min(platform_demand, taxis_here)
                unmatched = taxis_here - matched

                next_taxis[k, idx] += unmatched

                if matched > 0:
                    destinations = np.random.choice(self.n_arms, size=matched, p=sim_probs)
                    for dest in destinations:
                        next_taxis[k, dest] += 1

                        dest_pu = self.idx_to_pu[dest]
                        fare_info = self.graph_dict.get(pu, {}).get(dest_pu)
                        if fare_info:
                            base_fare = fare_info["fare"]
                        else:
                            base_fare = self.default_distance * self.price_per_mile + self.intercept

                        fare = base_fare * mults[k]
                        rewards[k, idx] += fare

        self.taxis = next_taxis
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        return self._get_states(), rewards, done, {}


def run_monopoly_experiment(args, latent_df, graph_dict, predictor, do_to_pu, default_fare_params):
    print("\n--- Running Monopoly (Independent Q-Learning) Simulation ---")
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
        intercept=default_fare_params["intercept"],
        active_multiplier=args.active_multiplier
    )

    # Use run_iql from iql
    q_tables, rewards_history = run_iql(
        env=env,
        n_agents=env.n_arms,
        n_states=args.n_states,
        n_actions=5,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        alpha=args.alpha,
        discount=args.discount,
        eps=args.eps
    )

    print(f"Monopoly Simulation completed over {args.episodes} episodes.")
    print(f"Final Episode total revenue: ${rewards_history[-1]:,.2f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label="Episode Revenue", color="#1f77b4", alpha=0.4)
    # Plot moving average
    window = min(10, len(rewards_history))
    moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(rewards_history)), moving_avg, label=f"{window}-Ep Moving Avg", color="#005b96", linewidth=2.5)
    
    plt.title("Monopoly Ride-Hailing Independent Q-Learning (IQL) Revenue Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Revenue ($)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_file = PROJECT_DIR / "data" / "monopoly_results.png"
    plt.savefig(out_file, dpi=300)
    print(f"Results plot saved to: [monopoly_results.png](file://{out_file})")
    return rewards_history


def run_oligopoly_experiment(args, latent_df, graph_dict, predictor, do_to_pu, default_fare_params):
    print("\n--- Running Oligopoly (4-Agent EXP3) Competition Simulation ---")
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

    multipliers = np.array([1.0, 1.2, 1.5, 1.8, 2.0])
    agents = [
        Exp3Agent(n_arms=env.n_arms, n_actions=4, gamma=args.gamma, eta=args.eta)
        for _ in range(4)
    ]

    n_agents = 4
    n_arms = env.n_arms
    agent_rewards_history = [[] for _ in range(n_agents)]

    for ep in range(args.episodes):
        states = env.reset()
        ep_rewards = np.zeros(n_agents)

        for step in range(args.steps):
            # Select action index and multiplier for each agent
            actions_indices = []
            actions_mults = []
            for k in range(n_agents):
                acts = agents[k].select_actions() # (n_arms,)
                actions_indices.append(acts)
                actions_mults.append(multipliers[acts])

            # Run environment step
            # env.step expects shape (4, n_arms)
            next_states, rewards, done, _ = env.step(np.array(actions_mults))
            # rewards has shape (4, n_arms)

            # Update weights for each EXP3 agent
            for k in range(n_agents):
                taxis = env.taxis[k]
                scaled_rewards = np.zeros(n_arms)
                for arm in range(n_arms):
                    n_taxis = taxis[arm]
                    if n_taxis > 0:
                        # Normalize reward per taxi to [0, 1] (conservative max fare per ride is 150 * 1.8 = 270)
                        avg_fare = rewards[k, arm] / n_taxis
                        scaled_rewards[arm] = np.clip(avg_fare / 270.0, 0.0, 1.0)
                    else:
                        scaled_rewards[arm] = 0.0

                agents[k].update(actions_indices[k], scaled_rewards)
                ep_rewards[k] += float(np.sum(rewards[k]))

            states = next_states
            if done:
                break

        for k in range(n_agents):
            agent_rewards_history[k].append(ep_rewards[k])

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep+1}/{args.episodes} | Total Revenue: "
                  f"P1: ${ep_rewards[0]:,.2f} | "
                  f"P2: ${ep_rewards[1]:,.2f} | "
                  f"P3: ${ep_rewards[2]:,.2f} | "
                  f"P4: ${ep_rewards[3]:,.2f}")

    print(f"Oligopoly Simulation completed.")
    for k in range(n_agents):
        print(f"Platform {k+1} final episode revenue: ${agent_rewards_history[k][-1]:,.2f}")

    # Plot results
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

    out_file = PROJECT_DIR / "data" / "oligopoly_results.png"
    plt.savefig(out_file, dpi=300)
    print(f"Results plot saved to: [oligopoly_results.png](file://{out_file})")
    return agent_rewards_history


def main():
    parser = argparse.ArgumentParser(description="RL Taxi Simulation Experiment Pipeline")
    parser.add_argument("--mode", type=str, choices=["monopoly", "oligopoly"], default="monopoly",
                        help="Choose monopoly simulation (RMAB Q-learning) or oligopoly simulation (4-Agent EXP3)")
    ### Set defaults for experiment configurations. n-states is set to 1 for state-free learning,
    ### and alpha is set to 0.2 to accelerate Q-value convergence under IQL.

    parser.add_argument("--episodes", type=int, default=80, help="Number of episodes to simulate")
    parser.add_argument("--steps", type=int, default=24, help="Time steps per episode (hours)")
    parser.add_argument("--taxis", type=int, default=5000, help="Total fleet size of taxis in the simulation")
    parser.add_argument("--n-states", type=int, default=1, help="Number of states for discretized taxi count")
    parser.add_argument("--bin-size", type=int, default=10, help="Bin size to discretize taxi counts")
    parser.add_argument("--theta", type=float, default=0.4, help="Passenger price sensitivity parameter")
    parser.add_argument("--discount", type=float, default=0.95, help="Discount factor (discount)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Monopoly-specific arguments
    parser.add_argument("--budget", type=int, default=80, help="Monopoly budget constraint on active (surging) zones")
    parser.add_argument("--alpha", type=float, default=0.2, help="Q-learning rate (alpha)")
    parser.add_argument("--eps", type=float, default=0.1, help="Q-learning exploration probability (eps)")
    parser.add_argument("--active-multiplier", type=float, default=1.5, help="Surge pricing multiplier for active Monopoly action") ## to modify to give a few more options

    # Oligopoly-specific arguments
    parser.add_argument("--gamma", type=float, default=0.1, help="EXP3 exploration parameter (gamma)")
    parser.add_argument("--eta", type=float, default=0.1, help="EXP3 updating step size (eta)")

    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)

    data_dir = PROJECT_DIR / "data"
    
    # Load required data
    print(f"Loading data from {data_dir}...")
    latent_df = pl.read_csv(data_dir / "latent_mle_params.csv")
    graph_df = pl.read_csv(data_dir / "graph.csv")
    
    # Load predictor model
    predictor_path = data_dir / "destination_predictor.pkl"
    with open(predictor_path, "rb") as f:
        predictor = pickle.load(f)

    # Establish dropoff to pickup mapping
    do_to_pu = {}
    for do_val in predictor.unique_do:
        if do_val in predictor.unique_pu:
            do_to_pu[do_val] = do_val
        else:
            # Fallback to the first element of unique_pu
            do_to_pu[do_val] = predictor.unique_pu[0]

    # Convert graph to a nested dictionary for fast lookups
    print("Building price graph lookup tables...")
    graph_dict = {}
    for row in graph_df.iter_rows(named=True):
        pu = row["PULocationID"]
        do = row["DOLocationID"]
        if pu not in graph_dict:
            graph_dict[pu] = {}
        graph_dict[pu][do] = {
            "distance": row["baseline_distance"],
            "time": row["baseline_time"],
            "fare": row["baseline_fare"]
        }

    # Train a baseline regression model for any missing edges in graph.csv
    x_reg = graph_df["baseline_distance"].to_numpy().reshape(-1, 1)
    y_reg = graph_df["baseline_fare"].to_numpy()
    reg_model = LinearRegression().fit(x_reg, y_reg)
    default_fare_params = {
        "price_per_mile": float(reg_model.coef_[0]),
        "intercept": float(reg_model.intercept_)
    }
    print(f"Graph loaded. Baseline fare estimator: fare = distance * {default_fare_params['price_per_mile']:.2f} + {default_fare_params['intercept']:.2f}")

    if args.mode == "monopoly":
        run_monopoly_experiment(args, latent_df, graph_dict, predictor, do_to_pu, default_fare_params)
    elif args.mode == "oligopoly":
        run_oligopoly_experiment(args, latent_df, graph_dict, predictor, do_to_pu, default_fare_params)


if __name__ == "__main__":
    main()