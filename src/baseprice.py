import polars as pl 
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def format_df(input, output): 
    """
    Basic preprocessing for FHVHV Dataset.
    """
    lf = pl.scan_parquet(input)
    processed_lf = (
        lf.with_columns([
            ((pl.col("pickup_datetime") - pl.col("request_datetime")).dt.total_seconds()).alias("waiting_time"), 
            (pl.col("request_datetime").dt.day()).alias("request_date"),
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

def establish_baseline(input, data_path): 
    """
    Calculates base prices ("edge weights") for each (start, destination) pair / each edge. 
    Uses Tuesday 2am - 4am as a baseline. 
    Here, base price for each (start, destination) pair is just the average price across all rides in this region.
    """
    lf = pl.scan_parquet(input)
    dist_time_matrix = (
        lf
        .group_by(['PULocationID', 'DOLocationID'])
        .agg(
            pl.col("trip_miles").mean().round(2).alias("baseline_distance"), #all edges will have this. 
            pl.col("trip_time").mean().round(2).alias("baseline_time"), 
        )
        .collect()
        .sort(['PULocationID', 'DOLocationID'])
    )
    relevant_lf = lf.filter((pl.col("day_of_week").is_in([2, 3])) & (pl.col("request_hour").is_in([2, 3])))
    baseline_matrix = (
        relevant_lf
        .group_by(['PULocationID', 'DOLocationID'])
        .agg(
            pl.col("trip_miles").mean().round(2).alias("baseline_distance"), #still needed to calculate regression.
            pl.col("base_fare").mean().round(2).alias("baseline_fare")
        )
        .collect()
        .sort(['PULocationID', 'DOLocationID'])
    )
    intercept, price_per_mile = regression(baseline_matrix) #needs both fare & distance

    final_graph = dist_time_matrix.join(baseline_matrix, on = ['PULocationID', 'DOLocationID'], how = 'full', coalesce = True).drop(['baseline_distance_right'])
    final_graph = final_graph.with_columns(
        pl.when(pl.col("baseline_fare").is_null())
        .then((pl.col("baseline_distance") * price_per_mile + intercept).round(2))
        .otherwise(pl.col("baseline_fare"))
        .alias("baseline_fare")
    )

    # cannot sort df columns in with_columns operator -- isolates columns and performs operations on them. 
    # row orders must be kept constant.
    final_graph = final_graph.sort(['PULocationID', 'DOLocationID']) 
    final_graph.write_csv(data_path / "graph.csv")
    print(f"Successfully retrieved baseline prices for every edge, graph created.")

def regression(df, xcol = "baseline_distance", ycol = "baseline_fare"): 
    """
    Simple linear regression to get a baseline price per mile. 
    Time not included to avoid collinearity. 
    Only for edges where there's no ride from region A to region B. 
    """
    x = df[xcol].to_numpy().reshape(-1, 1)
    y = df[ycol].to_numpy()
    
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # 2. Calculate the Root Mean Squared Error
    rmse = root_mean_squared_error(y, y_pred)
    price_per_mile = model.coef_[0]
    intercept = model.intercept_
    
    print(f"Baseline price per mile: {model.coef_[0]:.2f}")
    print(f"Model Intercept: {model.intercept_:.2f}")
    print(f"Model RMSE: {rmse:.2f}")
    return intercept, price_per_mile


if __name__ == "__main__": 
    # 1. Get the absolute path of the directory where this script lives
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # 2. Build the paths dynamically relative to the script location
    # This mimics the "../data/" structure safely
    data_path = SCRIPT_DIR.parent / "data" 
    
    # 3. Run the function
    format_df(str(data_path / "fhvhv_tripdata_2026-01.parquet"), str(data_path / "new.parquet"))
    establish_baseline(str(data_path / "new.parquet"), data_path)

"""
Remarks: 
- The regression has a RMSE of 9.23, which is approximately an average error of $9 per ride at 2-5am on Tuesdays and Wednesdays. 
- It's to be expected since we're taking an average of all rides across all regions in NYC at that time. 
- Variations across regions can't be captured with a SLR (time not included to avoid collinearity). 
"""