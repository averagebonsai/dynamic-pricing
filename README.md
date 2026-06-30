Online Learning for Dynamic Pricing and Inventory Management

STRATEGIC FLEET ALLOCATION.
- Outsmarting the NYC Oligopoly with Adversarial Bandits.
- Can competing RL-based ride-hailing firms learn to tacitly collude?

The entry point for this experiment is in src/experiment_pipeline.py. 
- It automatically runs the monopoly and the oligopoly experiments. 
- Choose how many episodes and steps you want. 
- An example call to run in terminal: .venv/bin/python3 src/experiment_pipeline.py --episodes 70 --steps 120

For more details on the assumptions, refer to the Lark document: https://qjpn0nnhnxfd.jp.larksuite.com/wiki/AIV8w1NvLiYWtbkFPk2jM77KpPd

OVERVIEW OF THE PROJECT:
- This project investigates how competing, reinforcement learning (RL)-based ride-hailing platforms (like Uber and Lyft) interact within the New York City duopoly/oligopoly market.
Specifically, it explores algorithmic collusion—whether independent AI agents managing different platforms will autonomously learn to coordinate, artificially keeping surge prices high to exploit passengers without any explicit human agreement, or if they will fall into a competitive price war.
Pivots & Shift in Focus
The project shifted its focus away from a purely theoretical comparison of "Monopoly vs. Oligopoly" profits. 
