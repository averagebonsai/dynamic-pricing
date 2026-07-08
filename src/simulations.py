import numpy as np
import polars as pl
from destination import DestinationNNPredictor


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
        self.prediction_cache = {}

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
        #assume states is [1, 2]. that means first zone has state 1, which COULD mean 0-4. second zone has state 2, which COULD mean 5-9. 

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        # Compute current time
        total_hours = self.start_hour + self.current_step
        hour = total_hours % 24
        day = (self.start_day + total_hours // 24) % 7

        # Compute next time for reallocation demand lookup
        next_total_hours = total_hours + 1
        next_hour = next_total_hours % 24
        next_day = (self.start_day + next_total_hours // 24) % 7

        # Takes next-step demand and reallocates idle taxis
        next_demands = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            pu_id = self.idx_to_pu[i] #from index to pick-up zone. they're not the same. 
            next_demands[i] = self.demand_lookup.get((pu_id, next_day, next_hour), 0.0) #if nothing, fallback = 0
        sum_next_demands = next_demands.sum()
        if sum_next_demands > 0:
            relo_probs = next_demands / sum_next_demands #probability distribution of where the taxis would go. 
        else:
            relo_probs = np.ones(self.n_arms) / self.n_arms #fallback: equal probability distribution of taxis across all arms. vectorised. 

        rewards = np.zeros(self.n_arms)
        next_taxis = np.zeros(self.n_arms, dtype=int) #initialise next round taxi allocations. 
        unmatched_count = 0

        for idx in range(self.n_arms): #iterate over every zone. note that index is sequential, PU zones are not, hence the mapping.

            # logic in these steps: multiplier chosen, demand calculated, send taxis, calculate rewards. 

            pu = self.idx_to_pu[idx]
            taxis_here = self.taxis[idx]
            action = actions[idx] #action becomes an index. 

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
            unmatched_count += unmatched

            if matched > 0: #note: still on a single zone here. 
                # Sample destinations
                cache_key = (pu, day, hour)
                if cache_key not in self.prediction_cache: #prevents recalculation -- if it's in cache, then just calls results from there. 
                    nn_probs = self.predictor.predict(pu, day, hour) #passenger distribution. 
                    sim_probs = np.zeros(self.n_arms) 
                    for do_idx, do_val in self.predictor.idx_to_do.items(): #each drop-off location has a index. 
                        target_pu = self.do_to_pu.get(do_val) # finds the equivalent pick-up point. same location expressed differently for pick-up/drop-off indexes.
                        if target_pu in self.pu_to_idx: 
                            sim_probs[self.pu_to_idx[target_pu]] += nn_probs[do_idx] # same data, but the indexes are different. instead of dropoff index, pickup index.
                    if sim_probs.sum() > 0:
                        sim_probs /= sim_probs.sum()
                    else:
                        sim_probs = np.ones(self.n_arms) / self.n_arms #equal probability across all states
                    self.prediction_cache[cache_key] = sim_probs
                else:
                    sim_probs = self.prediction_cache[cache_key]

                #destinations is an array of size matched that shows where each taxi went, governed by p = sim_probs. 
                #sim_probs instead of nn_probs because nn_probs follows drop-off destination indexes. we want pick-up destination indexes. <-
                destinations = np.random.choice(self.n_arms, size=matched, p=sim_probs) 

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

        # Reallocate unmatched (idle) taxis based on next period's normalized latent demand
        if unmatched_count > 0:
            relo_destinations = np.random.choice(self.n_arms, size=unmatched_count, p=relo_probs)
            dests, counts = np.unique(relo_destinations, return_counts=True)
            for dest, count in zip(dests, counts):
                next_taxis[dest] += count

        self.taxis = next_taxis #allocation of taxis at start of next period.
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
        self.prediction_cache = {}

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
        next_demands = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            pu_id = self.idx_to_pu[i]
            next_demands[i] = self.demand_lookup.get((pu_id, next_day, next_hour), 0.0)
        sum_next_demands = next_demands.sum()
        if sum_next_demands > 0:
            relo_probs = next_demands / sum_next_demands
        else:
            relo_probs = np.ones(self.n_arms) / self.n_arms

        rewards = np.zeros((4, self.n_arms))
        next_taxis = np.zeros((4, self.n_arms), dtype=int)
        unmatched_counts = np.zeros(4, dtype=int)

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
                p_accept = np.exp(-self.theta * (avg_mult - 1.0)) #higher price, lower probability
                
                probs = np.zeros(5)
                if sum_v > 0:
                    probs[:4] = p_accept * (v_vals / sum_v) #4 ride-hailing companies share p_accept (observed demand), softmax.
                else:
                    probs[:4] = p_accept * 0.25 #fallback: equal split of the passenger demand. 
                probs[4] = 1.0 - p_accept

                # Sample customer allocation
                demands = np.random.multinomial(D_latent, probs) #get demand for each ride-hailing company, in each area. 
            else:
                demands = np.zeros(5, dtype=int) #no demand in the area. 

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

            # Matching and relocation for each platform -- allocate all available taxis first. 
            for k in range(4):
                taxis_here = self.taxis[k, idx]
                platform_demand = demands[k]

                matched = min(platform_demand, taxis_here)
                unmatched = taxis_here - matched
                unmatched_counts[k] += unmatched

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

        # Afterwards, reallocate unmatched (idle) taxis based on next period's normalized latent demand
        for k in range(4):
            if unmatched_counts[k] > 0:
                relo_destinations = np.random.choice(self.n_arms, size=unmatched_counts[k], p=relo_probs)
                dests, counts = np.unique(relo_destinations, return_counts=True)
                for dest, count in zip(dests, counts):
                    next_taxis[k, dest] += count

        self.taxis = next_taxis
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        return self._get_states(), rewards, done, {}

