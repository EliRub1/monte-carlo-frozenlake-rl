

"""
Monte Carlo Experiments and Visualization
=========================================
This file contains functions for running experiments and creating visualizations.


Components:
- Experiment runners for different scenarios
- Plotting functions for all required graphs
- Analysis and comparison utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List

###TEST####
from scipy.ndimage import uniform_filter1d


# Import from main file
from mc_rl_project import (
    FrozenLakeEnvironment, MonteCarloAgent, 
    RandomPolicy, ScaledPolicy, AdaptivePolicy, 
    RecencyBufferedPolicy, DistanceAwarePolicy,
    train_mc_agent, run_multiple_trials
)


def smooth_curve(data, window_size=50):
    if window_size <= 1:
        return data
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().to_numpy()

import numpy as np
import random

SEED = 57
np.random.seed(SEED)
random.seed(SEED)



def plot_graph1_alpha_sweep(env: FrozenLakeEnvironment, alphas: List[float], 
                            epsilon: float, n_episodes: int, n_trials: int,
                            v_star_start: float = None):
    """
    Graph 1: Compare different α values for on-policy MC.
    
    On-policy MC is achieved by setting behaviour policy = target policy.
    We use AdaptivePolicy which updates to match the target policy.
    """
    plt.figure(figsize=(12, 7))
    
    # For on-policy: behaviour policy = target policy
    # We can use AdaptivePolicy since it adapts to the target
    
    all_results = {}
    
    for alpha in alphas:
        print(f"\nRunning α = {alpha}")
        
       
        results = run_multiple_trials(
            env, AdaptivePolicy, n_trials, n_episodes, alpha, epsilon
        )
        
        all_results[alpha] = results
        
        # Calculate mean and std
        mean_v = np.mean(results, axis=0)
        std_v = np.std(results, axis=0)
        
        # Smooth for better visualization - TY GOD IT FIXED
        mean_v_smooth = smooth_curve(mean_v)
        
        # Plot
        episodes = np.arange(n_episodes)
        plt.plot(episodes, mean_v_smooth, label=f'α = {alpha}', linewidth=2)
        
        # Add confidence interval
        plt.fill_between(episodes, 
                        smooth_curve(mean_v - 1.96 * std_v / np.sqrt(n_trials)),
                        smooth_curve(mean_v + 1.96 * std_v / np.sqrt(n_trials)),
                        alpha=0.2)
    
    
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('V(s_start)', fontsize=12)
    plt.title('Graph 1: On-Policy MC with Different α Values', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph1_alpha_sweep.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_results


def plot_graph3_behaviour_policies_alpha1(env: FrozenLakeEnvironment, 
                                         epsilon: float, n_episodes: int, 
                                         n_trials: int, v_star_start: float = None):
    """
    Graph 3: Compare all behaviour policies with α = 1.
    """
    plt.figure(figsize=(14, 7))
    
    alpha = 1.0
    
    # Define all behaviour policies
    policies = {
        'Random': (RandomPolicy, {}),
        'Scaled': (ScaledPolicy, {}),
        'Adaptive': (AdaptivePolicy, {}),
        'Recency-Buffered': (RecencyBufferedPolicy, {}),
        'Distance-Aware': (DistanceAwarePolicy, {})
    }
    
    all_results = {}
    
    for policy_name, (policy_class, policy_kwargs) in policies.items():
        print(f"\nRunning {policy_name} policy with α = {alpha}")
        
        results = run_multiple_trials(
            env, policy_class, n_trials, n_episodes, alpha, epsilon, **policy_kwargs
        )
        
        all_results[policy_name] = results
        
        # Calculate mean and std
        mean_v = np.mean(results, axis=0)
        std_v = np.std(results, axis=0)
        
        # Smooth
        mean_v_smooth = smooth_curve(mean_v)
        
        # Plot
        episodes = np.arange(n_episodes)
        plt.plot(episodes, mean_v_smooth, label=policy_name, linewidth=2)
        
        # Confidence interval
        plt.fill_between(episodes,
                        smooth_curve(mean_v - 1.96 * std_v / np.sqrt(n_trials)),
                        smooth_curve(mean_v + 1.96 * std_v / np.sqrt(n_trials)),
                        alpha=0.15)
    
    
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('V(s_start)', fontsize=12)
    plt.title('Graph 3: Off-Policy MC - Comparison of Behaviour Policies (α = 1)', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph3_behaviour_policies_alpha1.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_results


def plot_graph2_behaviour_policies_best_alpha(env: FrozenLakeEnvironment,
                                             best_alpha: float, epsilon: float,
                                             n_episodes: int, n_trials: int,
                                             v_star_start: float = None):
    """
    Graph 2 (BONUSSSS): Compare all behaviour policies with best α.
    """
    plt.figure(figsize=(14, 7))
    
    # Define all behaviour policies
    policies = {
        'Random': (RandomPolicy, {}),
        'Scaled': (ScaledPolicy, {}),
        'Adaptive': (AdaptivePolicy, {}),
        'Recency-Buffered': (RecencyBufferedPolicy, {}),
        'Distance-Aware': (DistanceAwarePolicy, {})
    }
    
    all_results = {}
    
    for policy_name, (policy_class, policy_kwargs) in policies.items():
        print(f"\nRunning {policy_name} policy with α = {best_alpha}")
        
        results = run_multiple_trials(
            env, policy_class, n_trials, n_episodes, best_alpha, epsilon, **policy_kwargs
        )
        
        all_results[policy_name] = results
        
        # Calculate mean and std
        mean_v = np.mean(results, axis=0)
        std_v = np.std(results, axis=0)
        
        # Smooth
        mean_v_smooth = smooth_curve(mean_v)
        
        # Plot
        episodes = np.arange(n_episodes)
        plt.plot(episodes, mean_v_smooth, label=policy_name, linewidth=2)
        
        # Confidence interval
        plt.fill_between(episodes,
                        smooth_curve(mean_v - 1.96 * std_v / np.sqrt(n_trials)),
                        smooth_curve(mean_v + 1.96 * std_v / np.sqrt(n_trials)),
                        alpha=0.15)
    
    
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('V(s_start)', fontsize=12)
    plt.title(f'Graph 2: Off-Policy MC - Comparison of Behaviour Policies (α = {best_alpha})',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'graph2_behaviour_policies_alpha{best_alpha}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_results


def plot_graph4_best_comparison(graph1_results: Dict, graph3_results: Dict,
                               graph2_results: Dict, best_alpha: float,
                               n_episodes: int, n_trials: int,
                               v_star_start: float = None):
    """
    Graph 4: Compare best on-policy vs best off-policy curves.
    """
    plt.figure(figsize=(12, 7))
    
    # Find best on-policy (from graph1)
    best_onpolicy_alpha = None
    best_onpolicy_final_v = -np.inf
    best_onpolicy_results = None
    
    for alpha, results in graph1_results.items():
        final_v = np.mean(results[:, -100:])  # Average of last 100 episodes
        if final_v > best_onpolicy_final_v:
            best_onpolicy_final_v = final_v
            best_onpolicy_alpha = alpha
            best_onpolicy_results = results
    
    # Find best off-policy from graph3 (α=1)
    best_offpolicy_alpha1_name = None
    best_offpolicy_alpha1_final_v = -np.inf
    best_offpolicy_alpha1_results = None
    
    for policy_name, results in graph3_results.items():
        final_v = np.mean(results[:, -100:])
        if final_v > best_offpolicy_alpha1_final_v:
            best_offpolicy_alpha1_final_v = final_v
            best_offpolicy_alpha1_name = policy_name
            best_offpolicy_alpha1_results = results
    
    # Find best off-policy from graph2 (best α)
    best_offpolicy_best_alpha_name = None
    best_offpolicy_best_alpha_final_v = -np.inf
    best_offpolicy_best_alpha_results = None
    
    if graph2_results:
        for policy_name, results in graph2_results.items():
            final_v = np.mean(results[:, -100:])
            if final_v > best_offpolicy_best_alpha_final_v:
                best_offpolicy_best_alpha_final_v = final_v
                best_offpolicy_best_alpha_name = policy_name
                best_offpolicy_best_alpha_results = results
    
    episodes = np.arange(n_episodes)
    
    # Plot best on-policy
    mean_v = np.mean(best_onpolicy_results, axis=0)
    std_v = np.std(best_onpolicy_results, axis=0)
    mean_v_smooth = smooth_curve(mean_v)
    
    plt.plot(episodes, mean_v_smooth, 
            label=f'Best On-Policy (α={best_onpolicy_alpha})',
            linewidth=2.5, color='blue')
    plt.fill_between(episodes,
                    smooth_curve(mean_v - 1.96 * std_v / np.sqrt(n_trials)),
                    smooth_curve(mean_v + 1.96 * std_v / np.sqrt(n_trials)),
                    alpha=0.2, color='blue')
    
    # Plot best off-policy with α=1
    mean_v = np.mean(best_offpolicy_alpha1_results, axis=0)
    std_v = np.std(best_offpolicy_alpha1_results, axis=0)
    mean_v_smooth = smooth_curve(mean_v)
    
    plt.plot(episodes, mean_v_smooth,
            label=f'Best Off-Policy α=1 ({best_offpolicy_alpha1_name})',
            linewidth=2.5, color='green')
    plt.fill_between(episodes,
                    smooth_curve(mean_v - 1.96 * std_v / np.sqrt(n_trials)),
                    smooth_curve(mean_v + 1.96 * std_v / np.sqrt(n_trials)),
                    alpha=0.2, color='green')
    
    # Plot best off-policy with best α 
    if best_offpolicy_best_alpha_results is not None:
        mean_v = np.mean(best_offpolicy_best_alpha_results, axis=0)
        std_v = np.std(best_offpolicy_best_alpha_results, axis=0)
        mean_v_smooth = smooth_curve(mean_v)
        
        plt.plot(episodes, mean_v_smooth,
                label=f'Best Off-Policy α={best_alpha} ({best_offpolicy_best_alpha_name})',
                linewidth=2.5, color='orange')
        plt.fill_between(episodes,
                        smooth_curve(mean_v - 1.96 * std_v / np.sqrt(n_trials)),
                        smooth_curve(mean_v + 1.96 * std_v / np.sqrt(n_trials)),
                        alpha=0.2, color='orange')
    
    
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('V(s_start)', fontsize=12)
    plt.title('Graph 4: Best On-Policy vs Best Off-Policy Comparison',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph4_best_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Best On-Policy: α={best_onpolicy_alpha}, Final V={best_onpolicy_final_v:.4f}")
    print(f"Best Off-Policy (α=1): {best_offpolicy_alpha1_name}, Final V={best_offpolicy_alpha1_final_v:.4f}")
    if best_offpolicy_best_alpha_results is not None:
        print(f"Best Off-Policy (α={best_alpha}): {best_offpolicy_best_alpha_name}, Final V={best_offpolicy_best_alpha_final_v:.4f}")
    


def visualize_policy(Q: np.ndarray, env: FrozenLakeEnvironment, title: str = "Policy"):
    """
    Visualize policy as arrows on the grid.
    """
    size = env.size
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid
    for i in range(size):
        for j in range(size):
            state = i * size + j
            cell_char = env.custom_map[i][j]
            
            # Set cell color
            if cell_char == 'S':
                color = 'lightgreen'
            elif cell_char == 'G':
                color = 'gold'
            elif cell_char == 'H':
                color = 'lightcoral'
            else:  # 'F'
                color = 'lightblue'
            
            rect = plt.Rectangle((j, size-1-i), 1, 1, 
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add arrow for best action
            if cell_char != 'H' and cell_char != 'G':
                best_action = np.argmax(Q[state])
                arrow = action_symbols[best_action]
                ax.text(j+0.5, size-1-i+0.5, arrow, 
                       ha='center', va='center', fontsize=20, fontweight='bold')
            
            # Add state value
            v_value = np.max(Q[state])
            ax.text(j+0.5, size-1-i+0.15, f'{v_value:.2f}',
                   ha='center', va='center', fontsize=8, color='darkblue')
    
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_value_heatmap(Q: np.ndarray, env: FrozenLakeEnvironment, title: str = "Value Function"):
    """
    Create heatmap of value function.
    """
    size = env.size
    V = np.max(Q, axis=1).reshape(size, size)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(V, annot=True, fmt='.3f', cmap='YlOrRd', 
                square=True, cbar_kws={'label': 'V(s)'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main For Test --- OKK
if __name__ == "__main__":
    print("Monte Carlo Experiments - Ready to Run")
    print("="*60)
    