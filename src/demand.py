import polars as pl 

def taxi_demand(input): 
    lf = pl.scan_parquet(input)
    
