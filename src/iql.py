import numpy as np
from typing import Any, Tuple

def run_iql(
    env: Any,
    n_agents: int,
    n_states: int,
    n_actions: int,
    episodes: int = 100,
    steps_per_episode: int = 24,
    alpha: float = 0.05,
    discount: float = 0.95,
    eps: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Independent Q-Learning (IQL) algorithm for taxi simulation.
    Each agent (pickup zone) maintains its own Q-table and learns to make pricing decisions.
    
    Parameters:
    -----------
    env:
        The taxi environment (e.g. MonopolyTaxiEnv).
    n_agents:
        Number of independent agents/zones.
    n_states:
        Number of discretized taxi states.
    n_actions:
        Number of actions (e.g. 2 for active/passive surge).
    episodes:
        Number of training episodes.
    steps_per_episode:
        Number of hours/time periods per episode.
    alpha:
        Learning rate.
    discount:
        Discount factor.
    eps:
        Exploration rate.
        
    Returns:
    --------
    q_tables:
        Numpy array of shape (n_agents, n_states, n_actions) containing the learned Q-values.
    episode_rewards:
        Numpy array of shape (episodes,) containing the total revenue per episode.
    """
    q_tables = np.zeros((n_agents, n_states, n_actions), dtype=float)
    episode_rewards = []
    
    for ep in range(episodes):
        states = np.asarray(env.reset(), dtype=int)
        total_reward = 0.0
        
        for step in range(steps_per_episode):
            actions = np.zeros(n_agents, dtype=int)
            for i in range(n_agents):
                # Epsilon-greedy action selection
                if np.random.rand() < eps:
                    actions[i] = np.random.randint(n_actions)
                else:
                    actions[i] = np.argmax(q_tables[i, states[i]])
                    
            next_states, rewards, done, _ = env.step(actions)
            next_states = np.asarray(next_states, dtype=int)
            rewards = np.asarray(rewards, dtype=float)
            
            # Tabular Q-learning update per agent
            for i in range(n_agents):
                s = states[i]
                a = actions[i]
                ns = next_states[i]
                r = rewards[i]
                
                target = r + discount * np.max(q_tables[i, ns])
                q_tables[i, s, a] += alpha * (target - q_tables[i, s, a])
                
            total_reward += float(np.sum(rewards))
            states = next_states
            
            if done:
                break
                
        episode_rewards.append(total_reward)
        
    return q_tables, np.asarray(episode_rewards)
