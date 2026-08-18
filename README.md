# Dynamic Pricing and Fleet Allocation

This repository contains the BlendED AI+X Project-Based Learning (PBL) Track 1 work on online learning for dynamic pricing and inventory management. It models ride-hailing demand in New York City and studies how learning-based platforms allocate vehicles and set surge multipliers under monopoly and competition.

## Research question

Can independent reinforcement-learning (RL) agents operating competing ride-hailing platforms learn pricing behaviour that resembles tacit coordination, or does competition produce a price war? The experiments compare:

- **Monopoly:** one platform uses independent tabular Q-learning (IQL) to choose a multiplier for each pickup zone.
- **Oligopoly:** four platforms use adversarial multi-armed bandits (EXP3) and compete for the same demand.

Demand is estimated from NYC For-Hire Vehicle High Volume (FHVHV) trip data. The simulation combines latent Poisson demand, destination-choice probabilities from a neural-network predictor, baseline origin-destination fares, passenger price sensitivity, and fleet reallocation between time steps.

## Repository layout

```text
src/
  environment.py                       Data preparation and demand estimation
  destination.py                       Destination prediction model and workflow
  simulations.py                       Monopoly and oligopoly environments
  rl_algorithms.py                     Generic Q-learning, SARSA, EXP3, and IQL utilities
  experiment_pipeline.py               Main monopoly/oligopoly experiment runner
  run_parallel_experiments.py          Parallel freeze/unfreeze experiment runner
  courthoud_*.py                       Freeze/unfreeze experiment variants
tests/
  test_fleet_size.py                   Fleet-size validation test
data/                                  Prepared inputs and trained predictor artifacts
experiments/                           Saved experiment outputs
archive/                               Archived algorithms and earlier work
```

## Requirements

- Python 3.10 or newer is recommended.
- Runtime dependencies are listed in [`requirements.txt`](requirements.txt): NumPy, Polars, pandas, scikit-learn, Matplotlib, `python-dotenv`, and `psycopg2-binary`.
- The experiment runner expects these prepared files in `data/`:
  - `latent_mle_params.csv`
  - `graph.csv`
  - `destination_predictor.pkl`

Create an isolated environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the main experiments

`experiment_pipeline.py` runs both the monopoly and four-platform oligopoly simulations. For a quick run:

```bash
python src/experiment_pipeline.py --episodes 70 --steps 120
```

Useful options include `--taxis` for fleet size, `--theta` for price sensitivity, `--seed` for reproducibility, and `--alpha`, `--eps`, `--gamma`, and `--eta` for the RL hyperparameters. Run `python src/experiment_pipeline.py --help` for the complete list.

The runner writes generated outputs under `tests/`:

- `tests/monopoly_results.png` and `tests/oligopoly_results.png` — revenue plots
- `tests/logs/monopoly_rewards_history.csv` — monopoly episode revenue
- `tests/logs/monopoly_multipliers_summary.csv` — aggregated monopoly multipliers
- `tests/logs/oligopoly_platform_<n>_summary.csv` — aggregated multipliers per platform

These are generated artifacts and may be overwritten by subsequent runs.

## Prepare data and train the destination model

The data workflow in `src/environment.py` transforms a raw FHVHV parquet file, estimates baseline fares, builds historical demand parameters, and estimates latent demand. It expects the raw file at `data/fhvhv_tripdata_2026-01.parquet` and writes `new.parquet`, `graph.csv`, `historical_mle_params.csv`, and `latent_mle_params.csv` to `data/`:

```bash
python src/environment.py
```

After `new.parquet` has been created, `src/destination.py` trains and saves the destination predictor and its model weights:

```bash
python src/destination.py
```

The default workflow samples up to 300,000 records, evaluates the model, and saves `data/destination_predictor.pkl`.

## Parallel experiments

To run the freeze/unfreeze variants across deviators and random seeds:

```bash
python src/run_parallel_experiments.py \
  --script courthoud_freeze_unfreeze_monopoly_reset.py \
  --deviators 0 1 2 3 \
  --seeds 42 43 \
  --cores 4 \
  --out-dir experiments/parallel_runs
```

Each run receives its own output directory and `simulation.log`. Use `--help` to see the forwarded episode, freeze, and evaluation controls.

## Tests

The fleet-size test uses the prepared files in `data/` and checks both scalar and per-platform fleet configurations:

```bash
python -m pytest tests
```

## Notes and assumptions

- Surge multipliers used by the main pipeline are `1.0`, `1.2`, `1.5`, `1.8`, and `2.0`.
- The current prepared `data/` artifacts contain **262 pickup zones** (as determined by the destination predictor's pickup-zone domain), not 252. The modular Q-learning table therefore has shape `(262, n_inventory_states, 5)`; with the default 10 inventory states, this is `(262, 10, 5)`.
- The default price-sensitivity parameter is `theta=0.4`, with acceptance model `exp(-theta * (multiplier - 1))`.
- The oligopoly baseline compares each platform with one quarter of the monopoly revenue.
- Results are simulations for research and education; they are not production pricing recommendations.

## License and attribution

This project was developed as part of BlendED's AI+X PBL programme. Add the project's intended license and data-use attribution here before public distribution.
