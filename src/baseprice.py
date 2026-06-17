import polars as pl 
from pathlib import Path

def format_df(input, output): 
    """
    Basic preprocessing for FHVHV Dataset.
    """
    lf = pl.scan_parquet(input)
    processed_lf = (
        lf.with_columns([
            ((pl.col("pickup_datetime") - pl.col("request_datetime")).dt.total_seconds()).alias("waiting_time"), 
            ((pl.col("request_datetime").dt.day() + 3) % 7).alias("day_of_week"),
            (pl.col("request_datetime").dt.hour()).alias("request_hour"),
            (pl.col("base_passenger_fare") + pl.col("bcf") + pl.col("sales_tax")).round(2).alias("base_fare"), 
            (pl.col("congestion_surcharge") + pl.col("cbd_congestion_fee")).alias("congestion_tax"), 
            (pl.col("driver_pay") + pl.col("tips")).round(2).alias("driver_income")
        ]).drop([
            "request_datetime", "on_scene_datetime", "pickup_datetime", "dropoff_datetime", "originating_base_num", 
            "base_passenger_fare", "bcf", "sales_tax", "congestion_surcharge", "cbd_congestion_fee", "driver_pay", "tips", 
            "shared_request_flag", "shared_match_flag", "access_a_ride_flag", "wav_request_flag", "wav_match_flag"
        ])
    ) 
    processed_lf.sink_parquet(output)
    print(f"Successfully exported LazyFrame to {output}")

def establish_baseline(input): 
    """
    Calculates base prices ("edge weights") for each (start, destination) pair / each edge. 
    Uses Tuesday 2am - 4am as a baseline. 
    Here, base price for each (start, destination) pair is just the average price across all rides in this region.
    """
    lf = pl.scan_parquet(input)
    relevant_lf = lf.filter((pl.col("day_of_week") == 2) & (pl.col("request_hour").is_in([2, 3])))
    baseline_matrix = (
        relevant_lf
        .group_by(['PULocationID', 'DOLocationID'])
        .agg(
            pl.col("base_fare").mean().round(2).alias("baseline_fare")
        )
        .collect()
        .sort(['PULocationID', 'DOLocationID'])
    )
    baseline_matrix.write_csv("baseline_matrix.csv")
    print(f"Successfully retrieved baseline prices for every edge.")


if __name__ == "__main__": 
    # 1. Get the absolute path of the directory where this script lives
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # 2. Build the paths dynamically relative to the script location
    # This mimics the "../data/" structure safely
    input_path = SCRIPT_DIR.parent / "data" / "fhvhv_tripdata_2026-01.parquet"
    output_path = SCRIPT_DIR.parent / "data" / "new.parquet"
    
    # 3. Run the function
    format_df(str(input_path), str(output_path))
    establish_baseline(str(output_path))






