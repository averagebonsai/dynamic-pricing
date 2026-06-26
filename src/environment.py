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
    dist_time_matrix = ( # across all time periods
        lf
        .group_by(['PULocationID', 'DOLocationID'])
        .agg(
            pl.col("trip_miles").mean().round(2).alias("baseline_distance"), #all edges will have this. 
            pl.col("trip_time").mean().round(2).alias("baseline_time"), 
        )
        .collect()
        .sort(['PULocationID', 'DOLocationID'])
    )
    relevant_lf = lf.filter((pl.col("day_of_week").is_in([2, 3])) & (pl.col("request_hour").is_in([2, 3]))) #only for tues/wed 2-3am
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
    return intercept, price_per_mile

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

def taxi_demand_historical(input, data_path): 
    lf = pl.scan_parquet(input)
    
    unique_dates = lf.select(["request_date", "day_of_week"]).unique()
    unique_zones = lf.select(pl.col("PULocationID").unique())

    grid_without_hours = unique_dates.join(unique_zones, how = 'cross')
    grid = grid_without_hours.with_columns(
        [pl.lit(list(range(24))).alias("request_hour")] #to account for rows with 0 rides. 
        ).explode("request_hour") #note: request_hour is here. if it were above, it won't have rows for quiet zones with no rides 

    historical_counts = ( 
        lf.group_by(["request_date", "day_of_week", "request_hour", "PULocationID"])
        .len(name = "num_trips") #number of rides in each date, request_hour, start location. day_of_week shouldn't have any impact
    )

    full_grid = (
        grid.join(
            historical_counts, 
            on = ["request_date", "day_of_week", "request_hour", "PULocationID"],
            how = 'left'
            )
            .with_columns(pl.col("num_trips").fill_null(0))
        ) #grid of counts for each unique time-place, including those with 0 rides

    mle_params = (
        full_grid.group_by(["PULocationID", "day_of_week", "request_hour"])
        .agg(pl.col("num_trips").mean().alias("historical_poisson")) #note: E(X) = lambda for Poisson Distribution.
        .sort(["PULocationID", "day_of_week", "request_hour"])
        )

    df = mle_params.collect()
    df.write_csv(data_path / "historical_mle_params.csv")
    print("Successfully written MLE parameters.") #note: this is historical demand, not latent demand (accounting for those who choose not to ride)

def latent_demand(input_parquet, graph_csv, historical_csv, intercept, output_path, theta = 0.4): 
    """
    1. Loads raw ride data and baseline graph.
    2. Calculates implied surge multipliers for every individual historical ride.
    3. Calculates the passenger acceptance probability for each ride.
    4. Aggregates by Origin & Hour of Week to compute the unsuppressed latent Poisson lambda.
    """

    lf = pl.scan_parquet(input_parquet)
    graph_df = pl.read_csv(graph_csv).select(['PULocationID', 'DOLocationID', 'baseline_distance', 'baseline_fare'])
    historical_df = pl.read_csv(historical_csv)

    joined_lf = lf.join(graph_df.lazy(), on = ['PULocationID', 'DOLocationID'], how = 'left')
    processed_lf = joined_lf.with_columns([
        pl.when(pl.col('baseline_fare') > 0)
        .then((((pl.col('base_fare') - intercept) / pl.col('trip_miles') * pl.col('baseline_distance') + intercept) / pl.col('baseline_fare')).clip(lower_bound = 1.0)) #implied surge multiplier: normalised fare price / baseline price
        .otherwise(1.0)
        .alias("implied_surge")
    ]).with_columns([
        ((-theta * (pl.col('implied_surge') - 1.0)).exp()).alias("p_accept") #probability of accepting the ride = exp(-theta * (surge - 1))
    ])

    init_poisson = (
        processed_lf
        .group_by(['PULocationID', 'day_of_week', 'request_hour'])
        .agg([
            pl.col('p_accept').mean().alias('avg_historical_acceptance')
        ])
        .collect()
        .sort(['PULocationID', 'day_of_week', 'request_hour'])
    )

    final_df = historical_df.join(init_poisson, on = ['PULocationID', 'day_of_week', 'request_hour'], how = 'left') #we're sure that the left df has a entry for every zone, every hour of week.
    final_df = final_df.with_columns(
        pl.when(pl.col("avg_historical_acceptance") > 0.01) #demand becomes wildly unstable if the denominator becomes 
        .then(pl.col("historical_poisson") / pl.col("avg_historical_acceptance")) 
        .otherwise(pl.col("historical_poisson")) 
        .round(2)
        .alias("latent_poisson") 
    )

    #note: here, we aggregate average historical acceptance across ALL 4 firms. latent x avg_historical_acceptance = historical/observed.
    #the difference between latent and historical is in the number of people overall who saw the price and chose not to take it. 
    #when we calculate how the taxi demand is spread across the 4 firms in competition, we use a different formula (Multinomial Logit Choice). 
    #there, choosing not to take any ride is considered an option, and is baked into the softmax equation. 

    final_df = final_df.with_columns(pl.col('avg_historical_acceptance').round(2))
    print(f"Minimum historical acceptance: {final_df['avg_historical_acceptance'].min()}") 

    final_df.write_csv(output_path / "latent_mle_params.csv")
    print("Successfully obtained latent poisson parameters.")


if __name__ == "__main__": 
    # 1. Get the absolute path of the directory where this script lives
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # 2. Build the paths dynamically relative to the script location
    # This mimics the "../data/" structure safely
    data_path = SCRIPT_DIR.parent / "data" 
    
    # 3. Run the function
    format_df(str(data_path / "fhvhv_tripdata_2026-01.parquet"), str(data_path / "new.parquet"))
    intercept, price_per_mile = establish_baseline(str(data_path / "new.parquet"), data_path)
    taxi_demand_historical(str(data_path / "new.parquet"), data_path)
    latent_demand(str(data_path / "new.parquet"), str(data_path / "graph.csv"), str(data_path / "historical_mle_params.csv"), intercept, data_path)

"""
Remarks for baseline price: 
- The regression has a RMSE of 9.23, which is approximately an average error of $9 per ride at 2-5am on Tuesdays and Wednesdays. 
- It's to be expected since we're taking an average of all rides across all regions in NYC at that time. 
- Variations across regions can't be captured with a SLR (time not included to avoid collinearity). 

Remarks for latent demand: 
- Underlying assumption: Surge multiplier and demand related by latent demand = historical demand / exp(-theta * (surge - 1))
"""
