"""Modular dynamic-pricing research pipeline.

Run the supported experiment entry point with:
    python -m src2.run_experiments --help
"""

from .config import PRICE_ACTIONS, LearningConfig, SimulationConfig

__all__ = ["PRICE_ACTIONS", "LearningConfig", "SimulationConfig"]
