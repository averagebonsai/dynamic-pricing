# Modular Dynamic-Pricing Pipeline

`src2` is the supported replacement for the legacy scripts in `src/`. It has one source of truth for pricing actions: `(1.0, 1.2, 1.5, 1.8, 2.0)`.

The current prepared `data/` artifacts contain 262 pickup zones.  Zone-level
Q-learning keeps a separate table for each zone, with default shape
`(262, 10, 5)`: 262 zones, 10 discretized taxi-inventory states, and five
surge-multiplier actions.  The zone count is loaded from the prepared
predictor rather than hard-coded by the simulation.

## Files

- `assets.py` — raw-trip preparation, fare and latent-demand estimation, destination model training, and runtime asset loading.
- `config.py` — validated simulation and learning configurations.
- `algorithms.py` — reusable zone-level Q-learning and EXP3 agents.
- `simulation.py` — one general environment for one or many competing firms.
- `experiments.py` — monopoly and oligopoly learning loops.
- `reporting.py` — CSV output and plots.
- `run_experiments.py` — supported CLI.

Run a prepared-data experiment from the repository root:

```bash
python -m src2.run_experiments --episodes 70 --steps 120
```

Build assets from the raw trip parquet before an initial run or after changing demand assumptions:

```bash
python -m src2.run_experiments --prepare-assets --theta 0.4
```
