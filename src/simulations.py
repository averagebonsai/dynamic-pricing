import numpy as np
import polars as pl
from destination import DestinationNNPredictor

# Shared global prediction cache to persist predictions across environment reinstantiations
_PREDICTION_CACHE = {}


class MonopolyTaxiEnv:
    """
    Environment wrapper for single-agent IQL simulation.
    Each taxi pickup zone is modeled as an arm in the bandit.
    The state of each arm is the discretized number of taxis currently in the zone.
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
        
        # Validate and parse fleet_size
        if isinstance(fleet_size, (list, np.ndarray, tuple)):
            if len(fleet_size) == 1:
                self.fleet_size = int(fleet_size[0])
            elif len(fleet_size) == 4:
                self.fleet_size = int(sum(fleet_size))
            else:
                raise ValueError("the number of inputs is not equal to the number of agents.")
        elif isinstance(fleet_size, (int, float, np.integer)):
            self.fleet_size = int(fleet_size)
        else:
            raise ValueError("the number of inputs is not equal to the number of agents.")
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

        self.taxis = np.zeros(self.n_arms, dtype=int) #initialise: array of 262 zeros. 
        self.current_step = 0
        self.prediction_cache = _PREDICTION_CACHE

        # --- Precomputations for Vectorization ---
        # Precompute the demand matrix of shape (n_arms, 7, 24)
        self.demand_matrix = np.zeros((self.n_arms, 7, 24), dtype=float)
        for i in range(self.n_arms):
            pu_id = self.idx_to_pu[i]
            for day_idx in range(7):
                for hr_idx in range(24):
                    self.demand_matrix[i, day_idx, hr_idx] = self.demand_lookup.get((pu_id, day_idx, hr_idx), 0.0)

        # Precompute the static mapping from destination index (do_idx) to pickup index (pu_idx)
        self.do_idx_to_pu_idx = np.empty(len(self.predictor.unique_do), dtype=int)
        for do_idx, do_val in self.predictor.idx_to_do.items():
            target_pu = self.do_to_pu.get(do_val)
            if target_pu in self.pu_to_idx:
                self.do_idx_to_pu_idx[do_idx] = self.pu_to_idx[target_pu]
            else:
                self.do_idx_to_pu_idx[do_idx] = 0

        # Precompute the fare matrix of shape (n_arms, n_arms)
        self.fare_matrix = np.empty((self.n_arms, self.n_arms), dtype=float)
        for i in range(self.n_arms):
            pu = self.idx_to_pu[i]
            for j in range(self.n_arms):
                dest_pu = self.idx_to_pu[j]
                fare_info = self.graph_dict.get(pu, {}).get(dest_pu)
                if fare_info:
                    self.fare_matrix[i, j] = fare_info["fare"]
                else:
                    self.fare_matrix[i, j] = self.default_distance * self.price_per_mile + self.intercept

    def reset(self) -> np.ndarray:
        self.current_step = 0

        # Distribute taxis uniformly across all zones
        probs = np.ones(self.n_arms) / self.n_arms 

        # Draw initial taxi counts
        self.taxis = np.random.multinomial(self.fleet_size, probs) #distributes 
        return self._get_states()

    def _get_states(self) -> np.ndarray:
        # Discretize taxi counts -- reduce number of states / strategies / arms, makes problem simpler to solve (small state space)
        states = np.minimum(self.taxis // self.bin_size, self.n_states - 1) #self.n_states - 1 --> duplicate the integer across len(self.taxis). 
        return states.astype(int) #an array of len(self.taxis), representing their discretised states. 

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        # Compute current time
        total_hours = self.start_hour + self.current_step
        hour = total_hours % 24
        day = (self.start_day + total_hours // 24) % 7

        # Compute next time for reallocation demand lookup
        next_total_hours = total_hours + 1
        next_hour = next_total_hours % 24
        next_day = (self.start_day + next_total_hours // 24) % 7

        # Vectorized next-step demand
        next_demands = self.demand_matrix[:, next_day, next_hour]
        sum_next_demands = next_demands.sum()
        if sum_next_demands > 0:
            relo_probs = next_demands / sum_next_demands
        else:
            relo_probs = np.ones(self.n_arms) / self.n_arms

        # Get batch predictions for destinations (cached across steps/episodes/environments)
        cache_key = (id(self.predictor), day, hour)
        if cache_key not in self.prediction_cache:
            pus = list(self.idx_to_pu.values())
            input_data = np.empty((len(pus), 3))
            input_data[:, 0] = pus
            input_data[:, 1] = day
            input_data[:, 2] = hour
            encoded_inputs = self.predictor.encoder.transform(input_data)
            batch_probs = self.predictor.model.predict_proba(encoded_inputs)
            
            sim_probs_batch = np.zeros((len(pus), self.n_arms))
            for do_idx in range(len(self.predictor.unique_do)):
                target_pu_idx = self.do_idx_to_pu_idx[do_idx]
                sim_probs_batch[:, target_pu_idx] += batch_probs[:, do_idx]
            
            row_sums = sim_probs_batch.sum(axis=1, keepdims=True)
            sim_probs_batch = np.where(row_sums > 0, sim_probs_batch / row_sums, 1.0 / self.n_arms)
            self.prediction_cache[cache_key] = sim_probs_batch
        else:
            sim_probs_batch = self.prediction_cache[cache_key]

        sim_probs_cdf = np.cumsum(sim_probs_batch, axis=1)

        rewards = np.zeros(self.n_arms)
        next_taxis = np.zeros(self.n_arms, dtype=int)

        # Sample Poisson demands for all zones in one vectorized call
        lambda_latents = self.demand_matrix[:, day, hour]
        # Multiplier choice
        multipliers_options = np.array([1.0, 1.2, 1.5, 1.8, 2.0], dtype=float)
        multipliers = multipliers_options[actions]
        
        lambda_actuals = lambda_latents * np.exp(-self.theta * (multipliers - 1.0))
        demands = np.random.poisson(lambda_actuals)

        matched = np.minimum(demands, self.taxis)
        unmatched = self.taxis - matched
        unmatched_count = unmatched.sum()

        zones_with_trips = np.where(matched > 0)[0]
        if len(zones_with_trips) > 0:
            origins = np.repeat(zones_with_trips, matched[zones_with_trips])
            M = len(origins)
            u = np.random.random(M)
            destinations = (sim_probs_cdf[origins] < u[:, np.newaxis]).sum(axis=1)
            destinations = np.minimum(destinations, self.n_arms - 1)
            
            next_taxis += np.bincount(destinations, minlength=self.n_arms)
            fares = self.fare_matrix[origins, destinations] * multipliers[origins]
            rewards += np.bincount(origins, weights=fares, minlength=self.n_arms)

        # Reallocate unmatched (idle) taxis based on next period's normalized latent demand
        if unmatched_count > 0:
            relo_destinations = np.random.choice(self.n_arms, size=unmatched_count, p=relo_probs)
            next_taxis += np.bincount(relo_destinations, minlength=self.n_arms)

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
        
        # Validate and parse fleet_size
        if isinstance(fleet_size, (list, np.ndarray, tuple)):
            if len(fleet_size) == 1:
                val = int(fleet_size[0])
                self.fleet_size = [val // 4] * 4
            elif len(fleet_size) == 4:
                self.fleet_size = [int(x) for x in fleet_size]
            else:
                raise ValueError("the number of inputs is not equal to the number of agents.")
        elif isinstance(fleet_size, (int, float, np.integer)):
            val = int(fleet_size)
            self.fleet_size = [val // 4] * 4
        else:
            raise ValueError("the number of inputs is not equal to the number of agents.")
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
        self.prediction_cache = _PREDICTION_CACHE

        # --- Precomputations for Vectorization ---
        # Precompute the demand matrix of shape (n_arms, 7, 24)
        self.demand_matrix = np.zeros((self.n_arms, 7, 24), dtype=float)
        for i in range(self.n_arms):
            pu_id = self.idx_to_pu[i]
            for day_idx in range(7):
                for hr_idx in range(24):
                    self.demand_matrix[i, day_idx, hr_idx] = self.demand_lookup.get((pu_id, day_idx, hr_idx), 0.0)

        # Precompute the static mapping from destination index (do_idx) to pickup index (pu_idx)
        self.do_idx_to_pu_idx = np.empty(len(self.predictor.unique_do), dtype=int)
        for do_idx, do_val in self.predictor.idx_to_do.items():
            target_pu = self.do_to_pu.get(do_val)
            if target_pu in self.pu_to_idx:
                self.do_idx_to_pu_idx[do_idx] = self.pu_to_idx[target_pu]
            else:
                self.do_idx_to_pu_idx[do_idx] = 0

        # Precompute the fare matrix of shape (n_arms, n_arms)
        self.fare_matrix = np.empty((self.n_arms, self.n_arms), dtype=float)
        for i in range(self.n_arms):
            pu = self.idx_to_pu[i]
            for j in range(self.n_arms):
                dest_pu = self.idx_to_pu[j]
                fare_info = self.graph_dict.get(pu, {}).get(dest_pu)
                if fare_info:
                    self.fare_matrix[i, j] = fare_info["fare"]
                else:
                    self.fare_matrix[i, j] = self.default_distance * self.price_per_mile + self.intercept

    def reset(self) -> np.ndarray:
        self.current_step = 0

        # Distribute taxis uniformly across all zones
        probs = np.ones(self.n_arms) / self.n_arms

        for k in range(4):
            self.taxis[k] = np.random.multinomial(self.fleet_size[k], probs)

        return self._get_states()

    def _get_states(self) -> np.ndarray:
        # Discretize taxi counts for each platform in each zone
        states = np.minimum(self.taxis // self.bin_size, self.n_states - 1) #unlike the monopoly, self.taxis is nested here. (4, 262)
        return states.astype(int) #number of taxis for each company in each zone

    def step(self, agent_multipliers: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        # agent_multipliers has shape (4, n_arms)
        total_hours = self.start_hour + self.current_step
        hour = total_hours % 24
        day = (self.start_day + total_hours // 24) % 7

        # Compute next time for reallocation demand lookup
        next_total_hours = total_hours + 1
        next_hour = next_total_hours % 24
        next_day = (self.start_day + next_total_hours // 24) % 7

        # Compute normalized next-step latent demand distribution for reallocation
        next_demands = self.demand_matrix[:, next_day, next_hour]
        sum_next_demands = next_demands.sum()
        if sum_next_demands > 0:
            relo_probs = next_demands / sum_next_demands
        else:
            relo_probs = np.ones(self.n_arms) / self.n_arms

        # Get batch predictions for destinations (cached across steps/episodes/environments)
        cache_key = (id(self.predictor), day, hour)
        if cache_key not in self.prediction_cache:
            pus = list(self.idx_to_pu.values())
            input_data = np.empty((len(pus), 3))
            input_data[:, 0] = pus
            input_data[:, 1] = day
            input_data[:, 2] = hour
            encoded_inputs = self.predictor.encoder.transform(input_data)
            batch_probs = self.predictor.model.predict_proba(encoded_inputs)
            
            sim_probs_batch = np.zeros((len(pus), self.n_arms))
            for do_idx in range(len(self.predictor.unique_do)):
                target_pu_idx = self.do_idx_to_pu_idx[do_idx]
                sim_probs_batch[:, target_pu_idx] += batch_probs[:, do_idx]
            
            row_sums = sim_probs_batch.sum(axis=1, keepdims=True)
            sim_probs_batch = np.where(row_sums > 0, sim_probs_batch / row_sums, 1.0 / self.n_arms)
            self.prediction_cache[cache_key] = sim_probs_batch
        else:
            sim_probs_batch = self.prediction_cache[cache_key]

        sim_probs_cdf = np.cumsum(sim_probs_batch, axis=1)

        rewards = np.zeros((4, self.n_arms))
        next_taxis = np.zeros((4, self.n_arms), dtype=int)

        # Sample Poisson demands for all zones in one vectorized call
        lambda_latents = self.demand_matrix[:, day, hour]
        D_latents = np.random.poisson(lambda_latents)

        # Compute choice probabilities for all zones in one step
        # agent_multipliers has shape (4, 262)
        v_vals = np.exp(-self.theta * (agent_multipliers - 1.0)) # shape (4, 262)
        sum_v = np.sum(v_vals, axis=0) # shape (262,)
        avg_mult = np.mean(agent_multipliers, axis=0) # shape (262,)
        p_accept = np.exp(-self.theta * (avg_mult - 1.0)) # shape (262,)

        probs = np.zeros((5, self.n_arms))
        valid = sum_v > 0
        probs[:4, valid] = p_accept[valid] * (v_vals[:, valid] / sum_v[valid])
        probs[:4, ~valid] = p_accept[~valid] * 0.25
        probs[4] = 1.0 - p_accept

        # Vectorized multinomial sampling for all active zones
        active_zones = np.where(D_latents > 0)[0]
        demands = np.zeros((5, self.n_arms), dtype=int)
        for idx in active_zones:
            demands[:, idx] = np.random.multinomial(D_latents[idx], probs[:, idx])

        # Vectorized matching
        matched = np.minimum(demands[:4], self.taxis) # shape (4, 262)
        unmatched = self.taxis - matched # shape (4, 262)
        unmatched_counts = unmatched.sum(axis=1) # shape (4,)

        # Vectorized destination choice and fare calculations per platform
        for k in range(4):
            matched_k = matched[k]
            zones_with_trips = np.where(matched_k > 0)[0]
            if len(zones_with_trips) > 0:
                origins = np.repeat(zones_with_trips, matched_k[zones_with_trips])
                M = len(origins)
                u = np.random.random(M)
                destinations = (sim_probs_cdf[origins] < u[:, np.newaxis]).sum(axis=1)
                destinations = np.minimum(destinations, self.n_arms - 1)
                
                next_taxis[k] += np.bincount(destinations, minlength=self.n_arms)
                fares = self.fare_matrix[origins, destinations] * agent_multipliers[k, origins]
                rewards[k] += np.bincount(origins, weights=fares, minlength=self.n_arms)

        # Reallocate unmatched (idle) taxis based on next period's normalized latent demand
        for k in range(4):
            if unmatched_counts[k] > 0:
                relo_destinations = np.random.choice(self.n_arms, size=unmatched_counts[k], p=relo_probs)
                next_taxis[k] += np.bincount(relo_destinations, minlength=self.n_arms)

        self.taxis = next_taxis
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        return self._get_states(), rewards, done, {}
