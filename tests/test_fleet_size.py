import sys
from pathlib import Path
import numpy as np
import polars as pl
import pickle

# Add project src to path dynamically
TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from destination import DestinationNNPredictor
sys.modules['__main__'].DestinationNNPredictor = DestinationNNPredictor
from simulations import MonopolyTaxiEnv, OligopolyTaxiEnv

def test_fleet_size():
    # Load minimal data to initialize environments
    data_dir = PROJECT_DIR / "data"
    latent_df = pl.read_csv(data_dir / "latent_mle_params.csv").head(10)
    
    with open(data_dir / "destination_predictor.pkl", "rb") as f:
        predictor = pickle.load(f)
        
    do_to_pu = {}
    for do_val in predictor.unique_do:
        if do_val in predictor.unique_pu:
            do_to_pu[do_val] = do_val
        else:
            do_to_pu[do_val] = predictor.unique_pu[0]
            
    graph_dict = {}
    
    print("Testing MonopolyTaxiEnv...")
    
    # 1. Single input as integer
    env = MonopolyTaxiEnv(
        latent_df=latent_df,
        graph_dict=graph_dict,
        predictor=predictor,
        do_to_pu=do_to_pu,
        fleet_size=5000
    )
    assert env.fleet_size == 5000, f"Expected 5000, got {env.fleet_size}"
    
    # 2. Single input as list
    env = MonopolyTaxiEnv(
        latent_df=latent_df,
        graph_dict=graph_dict,
        predictor=predictor,
        do_to_pu=do_to_pu,
        fleet_size=[5000]
    )
    assert env.fleet_size == 5000, f"Expected 5000, got {env.fleet_size}"
    
    # 3. Four inputs as list
    env = MonopolyTaxiEnv(
        latent_df=latent_df,
        graph_dict=graph_dict,
        predictor=predictor,
        do_to_pu=do_to_pu,
        fleet_size=[2900, 700, 700, 700]
    )
    assert env.fleet_size == 5000, f"Expected 5000, got {env.fleet_size}"
    
    # 4. Invalid input (e.g. length 2)
    try:
        MonopolyTaxiEnv(
            latent_df=latent_df,
            graph_dict=graph_dict,
            predictor=predictor,
            do_to_pu=do_to_pu,
            fleet_size=[2500, 2500]
        )
        raise AssertionError("MonopolyTaxiEnv failed to raise ValueError for 2 inputs")
    except ValueError as e:
        assert str(e) == "the number of inputs is not equal to the number of agents.", f"Unexpected error message: {e}"
        
    print("MonopolyTaxiEnv tests passed!")
    
    print("Testing OligopolyTaxiEnv...")
    
    # 1. Single input as integer
    env = OligopolyTaxiEnv(
        latent_df=latent_df,
        graph_dict=graph_dict,
        predictor=predictor,
        do_to_pu=do_to_pu,
        fleet_size=5000
    )
    assert env.fleet_size == [1250, 1250, 1250, 1250], f"Expected [1250, 1250, 1250, 1250], got {env.fleet_size}"
    
    # 2. Single input as list
    env = OligopolyTaxiEnv(
        latent_df=latent_df,
        graph_dict=graph_dict,
        predictor=predictor,
        do_to_pu=do_to_pu,
        fleet_size=[5000]
    )
    assert env.fleet_size == [1250, 1250, 1250, 1250], f"Expected [1250, 1250, 1250, 1250], got {env.fleet_size}"
    
    # 3. Four inputs as list
    env = OligopolyTaxiEnv(
        latent_df=latent_df,
        graph_dict=graph_dict,
        predictor=predictor,
        do_to_pu=do_to_pu,
        fleet_size=[2900, 700, 700, 700]
    )
    assert env.fleet_size == [2900, 700, 700, 700], f"Expected [2900, 700, 700, 700], got {env.fleet_size}"
    
    # 4. Invalid input (e.g. length 2)
    try:
        OligopolyTaxiEnv(
            latent_df=latent_df,
            graph_dict=graph_dict,
            predictor=predictor,
            do_to_pu=do_to_pu,
            fleet_size=[2500, 2500]
        )
        raise AssertionError("OligopolyTaxiEnv failed to raise ValueError for 2 inputs")
    except ValueError as e:
        assert str(e) == "the number of inputs is not equal to the number of agents.", f"Unexpected error message: {e}"
        
    print("OligopolyTaxiEnv tests passed!")

if __name__ == "__main__":
    test_fleet_size()
