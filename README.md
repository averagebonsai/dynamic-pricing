Here's the general gist of the environment set-up: 

1. Run python3 environment.py. This modifies the original fhvhv_tripdata_2026-01.parquet file, and outputs the following: 
- new.parquet (modified version of original data)
- graph.csv (graph mapping: baseline distance, time, price for each (origin, destination) pair)
- mle_params.csv (paassenger demand for each region and time-period)
- destination_predictor.pkl (a pickle file of MLP model weights -- to generate probability distribution of destination venues)

** Note: We need to simulate passengers requesting rides from an origin zone to a destination zone. The number of people that request for rides from a certain venue can be represented by a Non-Homogenous Poisson Process, given by λ. The destination venues are governed by a softmax distribution (i.e probabilities across all destinations sum to 1). By Poisson Splitting, the number of people that go from the origin to destination is also governed by a Non-Homogenous Poisson Process. 

** On implementation, when we determine how taxis are going to behave at each region (and each time period), we need to generate the probability distributions, and then send the corresponding number of taxis accordingly. For example, if the region has 40 passengers, and we have 3 zones with probabilities [0.6, 0.3, 0.1], then we would send 24, 12 and 4 taxis to each zone respectively. These probability distributions are to be generated dynamically at every time-step. (The alternative is a massive look-up table of roughly 11 million rows.)


What has not been done: 
- Something that makes passenger demand go down as prices go up. 
- The actual RL simulation (each step). 