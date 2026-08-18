"""Shared one-firm and multi-firm taxi-market simulation environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assets import SimulationAssets
from .config import SimulationConfig


@dataclass
class TaxiMarketEnv:
    """Stateful ride-hailing market with arbitrary number of competing firms.

    `step` accepts actual surge multipliers, rather than learner-specific action
    indexes.  This cleanly separates market economics from learning algorithms.
    """

    assets: SimulationAssets
    config: SimulationConfig
    n_firms: int
    rng: np.random.Generator
    fleet_sizes: np.ndarray = field(init=False)
    taxis: np.ndarray = field(init=False)  # (firm, zone)
    current_step: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.config.validate(self.n_firms)
        self.fleet_sizes = self.config.resolved_fleet_sizes(self.n_firms)
        self.taxis = np.zeros((self.n_firms, self.assets.n_zones), dtype=int)

    @property
    def n_zones(self) -> int:
        return self.assets.n_zones

    def reset(self) -> np.ndarray:
        """Start a new episode with each firm's fleet uniformly distributed."""
        self.current_step = 0
        uniform_zone_probability = np.full(self.n_zones, 1.0 / self.n_zones)
        for firm, fleet_size in enumerate(self.fleet_sizes):
            self.taxis[firm] = self.rng.multinomial(int(fleet_size), uniform_zone_probability)
        return self.inventory_states()

    def inventory_states(self) -> np.ndarray:
        """Discretize vehicle availability to keep tabular learning tractable."""
        return np.minimum(
            self.taxis // self.config.inventory_bin_size,
            self.config.n_inventory_states - 1,
        ).astype(int)

    def step(self, multipliers: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, np.ndarray | int]]:
        """Advance one hour and return inventory states plus zone-level revenue.

        The passenger-choice rule intentionally mirrors the legacy research
        model: overall acceptance responds to the mean platform multiplier,
        then accepted demand is allocated across firms by price attractiveness.
        """
        multipliers = np.asarray(multipliers, dtype=float)
        expected_shape = (self.n_firms, self.n_zones)
        if multipliers.shape != expected_shape:
            raise ValueError(f"multipliers must have shape {expected_shape}, got {multipliers.shape}.")
        if np.any(multipliers <= 0):
            raise ValueError("All surge multipliers must be positive.")

        day, hour = self._current_time()
        next_day, next_hour = self._next_time()
        starting_taxis = self.taxis.copy()
        latent_demand = self.assets.demand[:, day, hour]
        latent_arrivals = self.rng.poisson(latent_demand)

        customer_counts = self._allocate_customers(latent_arrivals, multipliers)
        served = np.minimum(customer_counts, starting_taxis)
        revenue, next_taxis = self._serve_and_move(served, multipliers, day, hour)
        self._relocate_idle_taxis(next_taxis, starting_taxis - served, next_day, next_hour)

        self.taxis = next_taxis
        self.current_step += 1
        done = self.current_step >= self.config.steps_per_episode
        info = {
            "demand": customer_counts,
            "served": served,
            "starting_taxis": starting_taxis,
            "ending_taxis": next_taxis.copy(),
            "day": day,
            "hour": hour,
        }
        return self.inventory_states(), revenue, done, info

    def _current_time(self) -> tuple[int, int]:
        total_hours = self.config.start_hour + self.current_step
        return (self.config.start_day + total_hours // 24) % 7, total_hours % 24

    def _next_time(self) -> tuple[int, int]:
        total_hours = self.config.start_hour + self.current_step + 1
        return (self.config.start_day + total_hours // 24) % 7, total_hours % 24

    def _allocate_customers(self, latent_arrivals: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
        """Sample platform demand plus an outside option for each origin zone."""
        attractiveness = np.exp(-self.config.price_sensitivity * (multipliers - 1.0))
        attractiveness_total = attractiveness.sum(axis=0)
        mean_multiplier = multipliers.mean(axis=0)
        acceptance_probability = np.exp(-self.config.price_sensitivity * (mean_multiplier - 1.0))
        probabilities = np.zeros((self.n_firms + 1, self.n_zones), dtype=float)
        valid = attractiveness_total > 0
        probabilities[: self.n_firms, valid] = (
            acceptance_probability[valid] * attractiveness[:, valid] / attractiveness_total[valid]
        )
        probabilities[: self.n_firms, ~valid] = acceptance_probability[~valid] / self.n_firms
        probabilities[-1] = 1.0 - acceptance_probability
        customer_counts = np.zeros((self.n_firms, self.n_zones), dtype=int)
        for zone in np.flatnonzero(latent_arrivals):
            customer_counts[:, zone] = self.rng.multinomial(int(latent_arrivals[zone]), probabilities[:, zone])[: self.n_firms]
        return customer_counts

    def _serve_and_move(self, served: np.ndarray, multipliers: np.ndarray, day: int, hour: int) -> tuple[np.ndarray, np.ndarray]:
        """Move taxis that served a rider and accumulate each origin's revenue."""
        destination_cdf = np.cumsum(self.assets.destination_probabilities(day, hour), axis=1)
        revenue = np.zeros((self.n_firms, self.n_zones), dtype=float)
        next_taxis = np.zeros((self.n_firms, self.n_zones), dtype=int)
        for firm in range(self.n_firms):
            origins = np.repeat(np.flatnonzero(served[firm]), served[firm, served[firm] > 0])
            if not len(origins):
                continue
            uniforms = self.rng.random(len(origins))
            destinations = (destination_cdf[origins] < uniforms[:, None]).sum(axis=1)
            destinations = np.minimum(destinations, self.n_zones - 1)
            next_taxis[firm] += np.bincount(destinations, minlength=self.n_zones)
            fares = self.assets.fares[origins, destinations] * multipliers[firm, origins]
            revenue[firm] += np.bincount(origins, weights=fares, minlength=self.n_zones)
        return revenue, next_taxis

    def _relocate_idle_taxis(self, next_taxis: np.ndarray, idle: np.ndarray, day: int, hour: int) -> None:
        """Reposition idle taxis toward the next hour's latent demand distribution."""
        next_demand = self.assets.demand[:, day, hour]
        probabilities = next_demand / next_demand.sum() if next_demand.sum() > 0 else np.full(self.n_zones, 1.0 / self.n_zones)
        for firm, idle_count in enumerate(idle.sum(axis=1)):
            if idle_count:
                destinations = self.rng.choice(self.n_zones, size=int(idle_count), p=probabilities)
                next_taxis[firm] += np.bincount(destinations, minlength=self.n_zones)
