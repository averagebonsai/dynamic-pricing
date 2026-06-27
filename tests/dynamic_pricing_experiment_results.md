# Dynamic Pricing Simulation: Monopoly vs. Oligopoly (120 Steps)

A comparison of taxi fleet pricing optimization under single-operator monopoly settings versus multi-platform competitive oligopoly settings, run over 120 steps per episode.

## 1. Monopoly Settings (Independent Q-Learning)
In this setting, a single operator controls the entire fleet of **5,000 taxis** and optimizes pricing multipliers across **262 pickup zones** using Independent Q-Learning (IQL).

### Configuration
* **Algorithm**: Independent Q-Learning (IQL)
* **Pricing Action Space**: 5 options `[1.0, 1.2, 1.5, 1.8, 2.0]`
* **Discretized States**: 1 (state-free configuration)
* **Learning Rate ($\alpha$)**: 0.2
* **Episodes**: 100
* **Steps per Episode (hours)**: 120
* **Final Episode Revenue**: **$4,637,327.06**
* **Average Chosen Pricing Multiplier**: **1.61**

### Monopoly Revenue Curve
![Monopoly IQL Revenue Curve](./monopoly_results.png)

---

## 2. Oligopoly Settings (4-Agent EXP3 Competition)
In this setting, 4 competing platforms split the fleet equally (**1,250 taxis each**) and compete for passengers using the EXP3 multi-armed bandit algorithm.

### Configuration
* **Algorithm**: 4-Agent EXP3
* **Pricing Action Space**: 5 options `[1.0, 1.2, 1.5, 1.8, 2.0]`
* **Episodes**: 100
* **Steps per Episode (hours)**: 120

### Results by Platform (Final Episode)
* **Platform 1**: $1,234,000.00 (approx)
* **Platform 2**: $1,240,000.00 (approx)
* **Platform 3**: $1,235,000.00 (approx)
* **Platform 4**: $1,241,572.16 (approx)
* **Total Combined Revenue**: **$4,950,572.16**
* **Average Chosen Pricing Multiplier**: **1.70**

### Oligopoly Revenue Comparison Curve
![Oligopoly EXP3 Revenue Curve](./oligopoly_results.png)

---

## 3. Analysis & Key Insights
* **Platform vs. Market View**: While the combined market revenue of the 4 competing Oligopoly platforms ($4.95M) is slightly higher than the Monopoly operator's revenue ($4.64M), the Monopoly operator earns **more than 3.7x** the revenue of any single competitor platform in the oligopoly setting (~$1.24M).
* **Dynamics under Longer Horizon**: Increasing the steps per episode to 120 (5 days of continuous operations) allows the spatial distribution of taxis to stabilize. The Monopoly operator settles on a slightly more conservative average multiplier of **1.61** to maintain higher taxi occupancy and throughput over the longer operational window.
