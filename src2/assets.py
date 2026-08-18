"""Offline data preparation, destination modelling, and runtime simulation assets.

This module is intentionally the only place that knows the input artifact
formats.  Experiment code consumes `SimulationAssets`, not CSVs or pickles.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder


@dataclass
class DestinationPredictor:
    """Predict a destination-zone probability vector from origin and time."""

    hidden_layer_sizes: tuple[int, ...] = (128, 64)
    max_epochs: int = 20
    random_state: int = 42
    encoder: OneHotEncoder | None = None
    model: MLPClassifier | None = None
    unique_pu: list[int] | None = None
    unique_dow: list[int] | None = None
    unique_hour: list[int] | None = None
    unique_do: list[int] | None = None
    do_to_idx: dict[int, int] | None = None
    idx_to_do: dict[int, int] | None = None

    def fit_encoder(self, frame: pl.DataFrame) -> None:
        """Freeze categorical domains before model training to keep shape stable."""
        self.unique_pu = sorted(frame["PULocationID"].unique().to_list())
        self.unique_dow = sorted(frame["day_of_week"].unique().to_list())
        self.unique_hour = sorted(frame["request_hour"].unique().to_list())
        self.unique_do = sorted(frame["DOLocationID"].unique().to_list())
        self.do_to_idx = {value: index for index, value in enumerate(self.unique_do)}
        self.idx_to_do = {index: value for value, index in self.do_to_idx.items()}
        self.encoder = OneHotEncoder(
            categories=[self.unique_pu, self.unique_dow, self.unique_hour],
            sparse_output=False,
            handle_unknown="ignore",
        )
        self.encoder.fit([[self.unique_pu[0], self.unique_dow[0], self.unique_hour[0]]])

    def _preprocess(self, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.encoder is None or self.do_to_idx is None:
            raise RuntimeError("Call fit_encoder before preprocessing data.")
        inputs = frame.select(["PULocationID", "day_of_week", "request_hour"]).to_numpy()
        targets = np.asarray([self.do_to_idx[value] for value in frame["DOLocationID"]], dtype=int)
        return self.encoder.transform(inputs), targets

    def fit(self, frame: pl.DataFrame) -> None:
        """Train the small MLP in reproducible mini-batches."""
        if self.encoder is None:
            self.fit_encoder(frame)
        features, targets = self._preprocess(frame)
        x_train, x_valid, y_train, y_valid = train_test_split(
            features, targets, test_size=0.15, random_state=self.random_state
        )
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            random_state=self.random_state,
        )
        classes = np.arange(len(self.unique_do or []))
        batch_size = 1024
        for epoch in range(self.max_epochs):
            permutation = np.random.default_rng(self.random_state + epoch).permutation(len(x_train))
            for start in range(0, len(x_train), batch_size):
                rows = permutation[start : start + batch_size]
                self.model.partial_fit(x_train[rows], y_train[rows], classes=classes)
            valid_loss = log_loss(y_valid, self.model.predict_proba(x_valid), labels=classes)
            print(f"destination epoch {epoch + 1}/{self.max_epochs}: validation log loss={valid_loss:.4f}")

    def predict_proba_batch(self, pickup_zones: np.ndarray, day: int, hour: int) -> np.ndarray:
        """Batch prediction is used by the simulator to avoid per-zone overhead."""
        if self.encoder is None or self.model is None:
            raise RuntimeError("Destination predictor is not fitted or loaded.")
        values = np.column_stack((pickup_zones, np.full(len(pickup_zones), day), np.full(len(pickup_zones), hour)))
        return self.model.predict_proba(self.encoder.transform(values))


# The alias lets this module load artifacts created by the legacy class name.
DestinationNNPredictor = DestinationPredictor


@dataclass(frozen=True)
class FareModel:
    """Fallback fare model for origin/destination pairs absent from graph.csv."""

    price_per_mile: float
    intercept: float


@dataclass
class SimulationAssets:
    """Prepared, read-mostly arrays required by the runtime simulator."""

    zone_ids: np.ndarray
    demand: np.ndarray  # (zone, day-of-week, hour), latent Poisson means
    fares: np.ndarray  # (origin zone, destination zone), base fares
    destination_to_zone: np.ndarray  # predictor destination class -> simulator zone
    predictor: DestinationPredictor
    fare_model: FareModel
    _destination_cache: dict[tuple[int, int], np.ndarray]

    @property
    def n_zones(self) -> int:
        return len(self.zone_ids)

    def destination_probabilities(self, day: int, hour: int) -> np.ndarray:
        """Return P(simulated destination zone | origin zone, day, hour)."""
        key = (int(day), int(hour))
        if key not in self._destination_cache:
            class_probabilities = self.predictor.predict_proba_batch(self.zone_ids, day, hour)
            zone_probabilities = np.zeros((self.n_zones, self.n_zones), dtype=float)
            for class_index, zone_index in enumerate(self.destination_to_zone):
                zone_probabilities[:, zone_index] += class_probabilities[:, class_index]
            row_totals = zone_probabilities.sum(axis=1, keepdims=True)
            self._destination_cache[key] = np.divide(
                zone_probabilities,
                row_totals,
                out=np.full_like(zone_probabilities, 1.0 / self.n_zones),
                where=row_totals > 0,
            )
        return self._destination_cache[key]


def format_trips(raw_path: Path, output_path: Path) -> None:
    """Create the compact trip table used by the fare, demand, and ML stages."""
    source = pl.scan_parquet(raw_path)
    (
        source.with_columns(
            [
                ((pl.col("pickup_datetime") - pl.col("request_datetime")).dt.total_seconds()).alias("waiting_time"),
                pl.col("request_datetime").dt.day().alias("request_date"),
                ((pl.col("request_datetime").dt.day() + 3) % 7).alias("day_of_week"),
                pl.col("request_datetime").dt.hour().alias("request_hour"),
                (pl.col("base_passenger_fare") + pl.col("bcf") + pl.col("sales_tax")).round(2).alias("base_fare"),
            ]
        )
        .drop(
            [
                "request_datetime", "on_scene_datetime", "pickup_datetime", "dropoff_datetime", "originating_base_num",
                "base_passenger_fare", "bcf", "sales_tax", "congestion_surcharge", "cbd_congestion_fee", "driver_pay", "tips",
                "shared_request_flag", "shared_match_flag", "access_a_ride_flag", "wav_request_flag", "wav_match_flag",
            ]
        )
        .sink_parquet(output_path)
    )


def fit_fare_graph(trips_path: Path, graph_path: Path) -> FareModel:
    """Estimate an OD base-fare graph and regress a fallback price from distance."""
    trips = pl.scan_parquet(trips_path)
    all_edges = trips.group_by(["PULocationID", "DOLocationID"]).agg(
        pl.col("trip_miles").mean().round(2).alias("baseline_distance"),
        pl.col("trip_time").mean().round(2).alias("baseline_time"),
    ).collect()
    quiet_edges = trips.filter(
        pl.col("day_of_week").is_in([2, 3]) & pl.col("request_hour").is_in([2, 3])
    ).group_by(["PULocationID", "DOLocationID"]).agg(
        pl.col("trip_miles").mean().round(2).alias("baseline_distance"),
        pl.col("base_fare").mean().round(2).alias("baseline_fare"),
    ).collect()
    regression = LinearRegression().fit(
        quiet_edges["baseline_distance"].to_numpy().reshape(-1, 1), quiet_edges["baseline_fare"].to_numpy()
    )
    fare_model = FareModel(float(regression.coef_[0]), float(regression.intercept_))
    graph = all_edges.join(quiet_edges, on=["PULocationID", "DOLocationID"], how="full", coalesce=True).drop("baseline_distance_right")
    graph = graph.with_columns(
        pl.when(pl.col("baseline_fare").is_null())
        .then((pl.col("baseline_distance") * fare_model.price_per_mile + fare_model.intercept).round(2))
        .otherwise(pl.col("baseline_fare"))
        .alias("baseline_fare")
    ).sort(["PULocationID", "DOLocationID"])
    graph.write_csv(graph_path)
    return fare_model


def estimate_historical_demand(trips_path: Path, output_path: Path) -> pl.DataFrame:
    """Write hourly pickup-zone Poisson means, including observed zero-demand periods."""
    trips = pl.scan_parquet(trips_path)
    dates = trips.select(["request_date", "day_of_week"]).unique()
    zones = trips.select(pl.col("PULocationID").unique())
    grid = dates.join(zones, how="cross").with_columns(pl.lit(list(range(24))).alias("request_hour")).explode("request_hour")
    observed = trips.group_by(["request_date", "day_of_week", "request_hour", "PULocationID"]).len(name="num_trips")
    historical = grid.join(observed, on=["request_date", "day_of_week", "request_hour", "PULocationID"], how="left").with_columns(
        pl.col("num_trips").fill_null(0)
    ).group_by(["PULocationID", "day_of_week", "request_hour"]).agg(
        pl.col("num_trips").mean().alias("historical_poisson")
    ).collect().sort(["PULocationID", "day_of_week", "request_hour"])
    historical.write_csv(output_path)
    return historical


def estimate_latent_demand(
    trips_path: Path,
    graph_path: Path,
    historical_path: Path,
    output_path: Path,
    fare_model: FareModel,
    theta: float = 0.4,
) -> None:
    """Estimate hourly Poisson latent demand by correcting observed trip counts.

    This intentionally retains the project's original normalised-fare equation:
    the observed fare is projected onto the OD baseline distance before its
    implied surge and acceptance probability are calculated.
    """
    trips = pl.scan_parquet(trips_path)
    graph = pl.read_csv(graph_path)
    historical = pl.read_csv(historical_path)
    # Historical fares imply a historical acceptance estimate at each zone/time.
    joined = trips.join(
        graph.lazy().select(["PULocationID", "DOLocationID", "baseline_distance", "baseline_fare"]),
        on=["PULocationID", "DOLocationID"],
        how="left",
    )
    implied = joined.with_columns(
        pl.when(pl.col("baseline_fare") > 0)
        .then(
            (
                ((pl.col("base_fare") - fare_model.intercept) / pl.col("trip_miles") * pl.col("baseline_distance") + fare_model.intercept)
                / pl.col("baseline_fare")
            ).clip(lower_bound=1.0)
        )
        .otherwise(1.0)
        .alias("implied_surge")
    ).with_columns((-theta * (pl.col("implied_surge") - 1.0)).exp().alias("p_accept"))
    acceptance = implied.group_by(["PULocationID", "day_of_week", "request_hour"]).agg(
        pl.col("p_accept").mean().alias("avg_historical_acceptance")
    ).collect()
    historical.join(acceptance, on=["PULocationID", "day_of_week", "request_hour"], how="left").with_columns(
        pl.when(pl.col("avg_historical_acceptance") > 0.01)
        .then(pl.col("historical_poisson") / pl.col("avg_historical_acceptance"))
        .otherwise(pl.col("historical_poisson"))
        .round(2)
        .alias("latent_poisson")
    ).sort(["PULocationID", "day_of_week", "request_hour"]).write_csv(output_path)


def prepare_assets(data_dir: Path, raw_filename: str = "fhvhv_tripdata_2026-01.parquet", theta: float = 0.4) -> None:
    """Run the complete offline preparation flow and train a destination model."""
    raw_path = data_dir / raw_filename
    trips_path = data_dir / "new.parquet"
    graph_path = data_dir / "graph.csv"
    historical_path = data_dir / "historical_mle_params.csv"
    latent_path = data_dir / "latent_mle_params.csv"
    format_trips(raw_path, trips_path)
    fare_model = fit_fare_graph(trips_path, graph_path)
    estimate_historical_demand(trips_path, historical_path)
    estimate_latent_demand(trips_path, graph_path, historical_path, latent_path, fare_model, theta=theta)
    train_destination_predictor(trips_path, data_dir / "destination_predictor.pkl")


def train_destination_predictor(trips_path: Path, output_path: Path, sample_size: int = 300_000) -> None:
    """Train and persist the destination model from prepared trip records."""
    frame = pl.read_parquet(trips_path)
    if sample_size < len(frame):
        frame = frame.sample(n=sample_size, seed=42)
    train_rows, test_rows = train_test_split(np.arange(len(frame)), test_size=0.15, random_state=42)
    predictor = DestinationPredictor()
    predictor.fit_encoder(frame)
    predictor.fit(frame[train_rows])
    test_features, test_targets = predictor._preprocess(frame[test_rows])
    test_loss = log_loss(test_targets, predictor.model.predict_proba(test_features), labels=np.arange(len(predictor.unique_do or [])))
    print(f"destination held-out log loss={test_loss:.4f}")
    with output_path.open("wb") as output_file:
        pickle.dump(predictor, output_file)


def _load_predictor(path: Path) -> DestinationPredictor:
    """Load new artifacts and legacy pickles created when destination.py was __main__."""
    setattr(sys.modules["__main__"], "DestinationNNPredictor", DestinationPredictor)
    with path.open("rb") as input_file:
        return pickle.load(input_file)


def load_simulation_assets(data_dir: Path) -> SimulationAssets:
    """Load files once and transform them into indexed arrays for fast simulation."""
    latent = pl.read_csv(data_dir / "latent_mle_params.csv")
    graph = pl.read_csv(data_dir / "graph.csv")
    predictor = _load_predictor(data_dir / "destination_predictor.pkl")
    zone_ids = np.asarray(predictor.unique_pu, dtype=int)
    zone_to_index = {zone: index for index, zone in enumerate(zone_ids)}
    n_zones = len(zone_ids)
    demand = np.zeros((n_zones, 7, 24), dtype=float)
    for row in latent.iter_rows(named=True):
        index = zone_to_index.get(int(row["PULocationID"]))
        if index is not None:
            value = row.get("latent_poisson")
            demand[index, int(row["day_of_week"]), int(row["request_hour"])] = 0.0 if value is None else float(value)
    regression = LinearRegression().fit(graph["baseline_distance"].to_numpy().reshape(-1, 1), graph["baseline_fare"].to_numpy())
    fare_model = FareModel(float(regression.coef_[0]), float(regression.intercept_))
    fallback_fare = fare_model.intercept + 5.0 * fare_model.price_per_mile
    fares = np.full((n_zones, n_zones), fallback_fare, dtype=float)
    for row in graph.iter_rows(named=True):
        origin = zone_to_index.get(int(row["PULocationID"]))
        destination = zone_to_index.get(int(row["DOLocationID"]))
        if origin is not None and destination is not None:
            fares[origin, destination] = float(row["baseline_fare"])
    # A destination that is not in the pickup domain is assigned to zone zero,
    # matching the legacy model's explicit fallback until a geographic map exists.
    destination_to_zone = np.asarray([zone_to_index.get(int(zone), 0) for zone in predictor.unique_do], dtype=int)
    return SimulationAssets(zone_ids, demand, fares, destination_to_zone, predictor, fare_model, {})
