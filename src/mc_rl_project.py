
"""
Monte Carlo Reinforcement Learning Project
==========================================
This file contains the main implementation of Monte Carlo algorithms for the Frozen Lake environment.

Components:
- Environment setup and utilities
- Behaviour policies (random, scaled, adaptive, recency-buffered, custom)
- Off-policy Monte Carlo with Importance Sampling
- Training and evaluation functions
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from typing import Dict, List, Tuple, Callable
import pandas as pd
from tqdm import tqdm

import random

def set_global_seed(seed: int): 
    np.random.seed(seed)
    random.seed(seed)



class FrozenLakeEnvironment:
    """
    Wrapper for Frozen Lake environment with custom configurations.
    """
    
    def __init__(self, size: int = 6,  success_rate: float = 0.7,seed: int =57):
        """
        Initialize the Frozen Lake environment.
        
        """
        self.size = size
        self.success_rate = success_rate
        self.seed= seed
        
        np.random.seed(self.seed)
        
        # Generate custom map with specified hole density
        self.custom_map = self._generate_custom_map()
        
        # Create environment
        self.env = gym.make(
            'FrozenLake-v1',
            desc=self.custom_map,
            is_slippery=True,
            render_mode=None
        )
        
        self.env.reset(seed=self.seed)
        
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.start_state = 0
        

        
        print(">>> BEFORE modify_transition_probabilities")
        self._modify_transition_probabilities()

        print(">>> BEFORE modify_rewards")
        self._modify_rewards()

        print(">>> AFTER environment init")

        
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.start_state = 0
        
    def _generate_custom_map(self):
        return [
            "SFFFFF",
            "FHFHFF",
            "FFFHFF",
            "HFHFFF",
            "FFFHFH",
            "FFFFFG"
            ]

    
    def _modify_transition_probabilities(self):
        """Modify transition probabilities to match success_rate."""
        P = self.env.unwrapped.P
        
        for state in range(self.n_states):
            for action in range(self.n_actions):
                transitions = P[state][action]
                
                # Find the intended direction
                # In Frozen Lake: 0=Left, 1=Down, 2=Right, 3=Up
                new_transitions = []
                
                for prob, next_state, reward, done in transitions:
                    # The first transition is usually the intended one
                    if len(new_transitions) == 0:
                        # Intended direction gets success_rate probability
                        new_prob = self.success_rate
                    else:
                        # Other directions share remaining probability
                        new_prob = (1 - self.success_rate) / (len(transitions) - 1)
                    
                    new_transitions.append((new_prob, next_state, reward, done))
                
                P[state][action] = new_transitions
    
    def _modify_rewards(self):
        """Modify rewards so that falling into a hole gives -0.01."""
        P = self.env.unwrapped.P
        
        for state in range(self.n_states):
            for action in range(self.n_actions):
                transitions = P[state][action]
                new_transitions = []
                
                for prob, next_state, reward, done in transitions:
                    # Check if next_state is a hole
                    row = next_state // self.size
                    col = next_state % self.size
                    if self.custom_map[row][col] == 'H':
                        reward = -0.01
                    
                    new_transitions.append((prob, next_state, reward, done))
                
                P[state][action] = new_transitions
    
    def get_map_string(self) -> str:
        """Get string representation of the map."""
        return '\n'.join(self.custom_map)
    
    def reset(self):
        """Reset the environment."""
        return self.env.reset()
    
    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)
    
    def get_dynamics(self):
        """Get transition dynamics."""
        return self.env.unwrapped.P


class BehaviourPolicy:
    """Base class for behaviour policies."""
    
    def __init__(self, n_states: int, n_actions: int, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
    
    def get_action_probabilities(self, state: int, Q: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a given state.
        
        Args:
            state: Current state
            Q: Q-value table
            
        Returns:
            Array of action probabilities
        """
        raise NotImplementedError
    
    def sample_action(self, state: int, Q: np.ndarray) -> int:
        """Sample an action from the policy."""
        probs = self.get_action_probabilities(state, Q)
        return np.random.choice(self.n_actions, p=probs)


class RandomPolicy(BehaviourPolicy):
    """Uniform random policy - chooses all actions uniformly."""
    
    def get_action_probabilities(self, state: int, Q: np.ndarray) -> np.ndarray:
        return np.ones(self.n_actions) / self.n_actions


class ScaledPolicy(BehaviourPolicy): ## fixed againnnn
    """
    Scaled target policy
    where p_s ~ Uniform[0,1] for each state.
    """
    
    def __init__(self, n_states: int, n_actions: int, epsilon: float = 0.1):
        super().__init__(n_states, n_actions, epsilon)
        # Sample p_s for each state once
        self.p_s = np.random.uniform(0, 1, size=n_states)
    
    def get_action_probabilities(self, state: int, Q: np.ndarray) -> np.ndarray:
        # Get epsilon-greedy target policy
        target_probs = self._epsilon_greedy_probs(state, Q)
        
        # Apply scaling
        p_s = self.p_s[state]
        #p_s=np.random.uniform(0,1)
        scaled_probs = p_s * target_probs + (1 - p_s) / self.n_actions
        
        return scaled_probs
    
    def _epsilon_greedy_probs(self, state: int, Q: np.ndarray) -> np.ndarray:
        """Get epsilon-greedy probabilities for target policy."""
        probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
        best_action = np.argmax(Q[state])
        probs[best_action] += (1 - self.epsilon)
        return probs


class AdaptivePolicy(BehaviourPolicy):
    """
    Adaptive policy: starts with equiprobable, then uses previous policy estimate.
    (the previous guess of optimal policy)
    """
    
    def __init__(self, n_states: int, n_actions: int, epsilon: float = 0.1):
        super().__init__(n_states, n_actions, epsilon)
        # Start with equiprobable policy
        self.current_policy = np.ones((n_states, n_actions)) / n_actions
    
    def get_action_probabilities(self, state: int, Q: np.ndarray) -> np.ndarray:
        return self.current_policy[state]
    
    def update_policy(self, Q: np.ndarray):
        """Update the policy based on current Q-values (epsilon-greedy)."""
        for state in range(self.n_states):
            probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
            best_action = np.argmax(Q[state])
            probs[best_action] += (1 - self.epsilon)
            self.current_policy[state] = probs


class RecencyBufferedPolicy(BehaviourPolicy):
    """
    Recency buffered policy: maintains buffer of last 10 policies.
    At each state, samples a policy from buffer uniformly and uses it.
    """
    
    def __init__(self, n_states: int, n_actions: int, epsilon: float = 0.1, buffer_size: int = 10):
        super().__init__(n_states, n_actions, epsilon)
        self.buffer_size = buffer_size
        
        # Initialize buffer with equiprobable policy
        equiprobable = np.ones((n_states, n_actions)) / n_actions
        self.policy_buffer = deque([equiprobable.copy() for _ in range(buffer_size)], maxlen=buffer_size)
    
    def get_action_probabilities(self, state: int, Q: np.ndarray) -> np.ndarray:
        # Sample a policy from buffer uniformly
        selected_policy_idx = np.random.randint(0, len(self.policy_buffer))
        selected_policy = self.policy_buffer[selected_policy_idx]
        return selected_policy[state]
    
    def update_policy(self, Q: np.ndarray):
        """Add new epsilon-greedy policy to buffer."""
        new_policy = np.zeros((self.n_states, self.n_actions))
        for state in range(self.n_states):
            probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
            best_action = np.argmax(Q[state])
            probs[best_action] += (1 - self.epsilon)
            new_policy[state] = probs
        
        self.policy_buffer.append(new_policy)


class DistanceAwarePolicy(BehaviourPolicy): ###### EXPLAIN THAT THIS IS NOT 100% OFF POLICYYYYYY
    """
    Behaviour policy with state-dependent exploration based on distance to goal.
    Exploration is higher when the agent is far from the goal and lower when close.
    """

    def __init__(self, n_states: int, n_actions: int,
                 epsilon: float = 0.5, epsilon_min: float = 0.1):
        super().__init__(n_states, n_actions, epsilon)
        self.epsilon_min = epsilon_min
        self.grid_size = int(np.sqrt(n_states))
        self.max_dist = 2 * (self.grid_size - 1)

    def get_action_probabilities(self, state: int, Q: np.ndarray) -> np.ndarray:
        # Convert state index to grid coordinates
        row, col = divmod(state, self.grid_size)

        # Manhattan distance to goal (bottom-right corner)
        dist = (self.grid_size - 1 - row) + (self.grid_size - 1 - col)

        # State-dependent exploration rate
        epsilon_s = max(self.epsilon_min,
                        self.epsilon * dist / self.max_dist)

        # Epsilon-greedy distribution
        probs = np.ones(self.n_actions) * (epsilon_s / self.n_actions)
        best_action = np.argmax(Q[state])
        probs[best_action] += 1.0 - epsilon_s

        return probs


class MonteCarloAgent:
    """
    Off-policy Monte Carlo agent with importance sampling and scaling factor α.
    """
    
    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.99, 
                 alpha: float = 1.0, epsilon: float = 0.1):
        """
        Initialize MC agent.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            gamma: Discount factor
            alpha: Learning rate / scaling factor
            epsilon: Epsilon for target policy (epsilon-greedy)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Initialize Q-values
        self.Q = np.zeros((n_states, n_actions))
        
        # Cumulative sum of weights for each state-action pair
        self.C = np.zeros((n_states, n_actions))
    
    def get_epsilon_greedy_action(self, state: int) -> int:
        """Get action using epsilon-greedy policy (target policy)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def get_target_policy_prob(self, state: int, action: int) -> float:
        """Get probability of action under target policy (epsilon-greedy)."""
        best_action = np.argmax(self.Q[state])
        if action == best_action:
            return 1 - self.epsilon + self.epsilon / self.n_actions
        else:
            return self.epsilon / self.n_actions
    
    def generate_episode(self, env, behaviour_policy):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = behaviour_policy.sample_action(state, self.Q)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode.append((state, action, reward))
            state = next_state

        return episode


    
    def update_q_values(self, episode: List[Tuple[int, int, float]], 
                       behaviour_policy: BehaviourPolicy):
        """
        Update Q-values using off-policy MC with importance sampling.
        
        Update rule: Q(S_t, A_t) = Q(S_t, A_t) + α * (W / C(S_t, A_t)) * [G - Q(S_t, A_t)]
        """
        G = 0  # Return
        W = 1  # Importance sampling ratio
        
        # Process episode backwards
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            
            # Update return
            G = self.gamma * G + reward
            
            # Update cumulative sum of weights
            self.C[state, action] += W
            
            # Update Q-value with scaling factor α
            if self.C[state, action] > 0:
                self.Q[state, action] += self.alpha * (W / self.C[state, action]) * (G - self.Q[state, action])
            
            # Update importance sampling ratio
            target_prob = self.get_target_policy_prob(state, action)
            behaviour_prob = behaviour_policy.get_action_probabilities(state, self.Q)[action]
            
            if behaviour_prob == 0:
                break  # Episode doesn't contribute further
            
            W *= target_prob / behaviour_prob
            
            if W == 0:
                break
    
    def get_v_value(self, state: int) -> float:
        """Get V(s) = max_a Q(s,a)."""
        return np.max(self.Q[state])


def train_mc_agent(env: FrozenLakeEnvironment, behaviour_policy: BehaviourPolicy,
                   n_episodes: int, alpha: float, epsilon: float,
                   gamma: float = 0.99) -> Tuple[np.ndarray, List[float]]:
    print(">>> train_mc_agent STARTED")

    """
    Train an MC agent and return Q-values and V(s_start) over episodes.
    
    Args:
        env: Environment
        behaviour_policy: Behaviour policy to use
        n_episodes: Number of episodes to train
        alpha: Learning rate
        epsilon: Epsilon for target policy
        gamma: Discount factor
        
    Returns:
        Final Q-values and list of V(s_start) values over episodes
    """
    agent = MonteCarloAgent(env.n_states, env.n_actions, gamma, alpha, epsilon)
    v_start_history = []
    
    for episode_idx in range(n_episodes):
        # Generate episode
        if episode_idx == 0:
            print(">>> First episode") # DEBUGGGGGGGGGGGGGGGGGGGGGGGG HEREEEEEEEEEEEEEEE
        episode = agent.generate_episode(env, behaviour_policy)
        
        # Update Q-values
        agent.update_q_values(episode, behaviour_policy)
        
        # Update behaviour policy if needed (for adaptive and recency-buffered)
        if hasattr(behaviour_policy, 'update_policy'):
            behaviour_policy.update_policy(agent.Q)
        
        # Record V(s_start)
        v_start_history.append(agent.get_v_value(env.start_state))
    
    return agent.Q, v_start_history


def run_multiple_trials(env: FrozenLakeEnvironment, behaviour_policy_class,
                       n_trials: int, n_episodes: int, alpha: float, epsilon: float,
                       gamma: float = 0.99, **policy_kwargs) -> np.ndarray:
    """
    Run multiple independent trials and return V(s_start) trajectories.
    
    Returns:
        Array of shape (n_trials, n_episodes) containing V(s_start) values
    """
    
    BASE_SEED=57
    
    results = np.zeros((n_trials, n_episodes))
    
    for trial in tqdm(range(n_trials), desc=f"Running trials"):
        
        np.random.seed(BASE_SEED + trial)
        random.seed(BASE_SEED + trial)
        
        env.env.reset(seed=BASE_SEED + trial)
        
        
        # Create fresh behaviour policy for each trial
        behaviour_policy = behaviour_policy_class(env.n_states, env.n_actions, epsilon, **policy_kwargs)
        
        # Train agent
        _, v_start_history = train_mc_agent(env, behaviour_policy, n_episodes, alpha, epsilon, gamma)
        
        results[trial] = v_start_history
    
    return results


# Main execution
if __name__ == "__main__":
    print("Monte Carlo RL Project - Implementation Ready")
    print("=" * 50)
    
    SEED = 57
    set_global_seed(SEED)

    env = FrozenLakeEnvironment(size=6, success_rate=0.7, seed=SEED)
    # Create environment
    #env = FrozenLakeEnvironment(size=6, success_rate=0.7)
    print(f"\nEnvironment Map:\n{env.get_map_string()}")
    print(f"\nNumber of states: {env.n_states}")
    print(f"Number of actions: {env.n_actions}")