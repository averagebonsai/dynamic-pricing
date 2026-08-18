"""Result persistence and plots, deliberately independent from learning loops."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Experiments frequently run headless (for example on GCP or CI).  Select the
# file-only backend before importing pyplot so plotting never requires a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def save_episode_results(records: list[dict], output_dir: Path, market_name: str) -> Path:
    """Write a tidy one-row-per-episode/per-firm result table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{market_name}_episode_results.csv"
    pl.DataFrame(records).write_csv(path)
    return path


def save_multiplier_summary(multiplier_history: np.ndarray, zone_ids: np.ndarray, output_dir: Path, market_name: str) -> Path:
    """Save mean multiplier per firm and zone over the full training run."""
    # History has shape (episode, step, firm, zone), so average over episode/step.
    averages = multiplier_history.mean(axis=(0, 1))
    rows: list[dict] = []
    for firm in range(averages.shape[0]):
        for zone, multiplier in enumerate(averages[firm]):
            rows.append({"firm_id": firm, "PULocationID": int(zone_ids[zone]), "mean_multiplier": float(multiplier)})
    path = output_dir / f"{market_name}_multiplier_summary.csv"
    pl.DataFrame(rows).write_csv(path)
    return path


def plot_episode_revenue(records: list[dict], output_dir: Path, market_name: str) -> Path:
    """Plot firm-level episode revenue using the same tidy records as CSV output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(records)
    figure, axis = plt.subplots(figsize=(10, 6))
    for firm in sorted(frame["firm_id"].unique().to_list()):
        firm_frame = frame.filter(pl.col("firm_id") == firm).sort("episode")
        axis.plot(firm_frame["episode"], firm_frame["revenue"], alpha=0.7, label=f"Firm {firm + 1}")
    axis.set(title=f"{market_name.title()} episode revenue", xlabel="Episode", ylabel="Revenue ($)")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    path = output_dir / f"{market_name}_revenue.png"
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path
