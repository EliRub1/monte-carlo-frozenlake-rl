

"""
Main Runner for Monte Carlo RL Project
======================================
This script runs all experiments and generates all required plots. - run this file only.

Output:
    - All required graphs (graph1, graph3, graph4)
    - Policy visualizations
    - Value function heatmaps
    - Summary statistics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # For saving plots without display
import matplotlib.pyplot as plt

from mc_rl_project import (
    FrozenLakeEnvironment, MonteCarloAgent,
    RandomPolicy, ScaledPolicy, AdaptivePolicy,
    RecencyBufferedPolicy, DistanceAwarePolicy,
    train_mc_agent
)

from mc_experiments import (
    plot_graph1_alpha_sweep,
    plot_graph2_behaviour_policies_best_alpha,
    plot_graph3_behaviour_policies_alpha1,
    plot_graph4_best_comparison,
    visualize_policy,
    create_value_heatmap
)

import random


def main():
    """Main execution function."""
    
    print("="*70)
    print("MONTE CARLO REINFORCEMENT LEARNING PROJECT")
    print("="*70)
    
    # ********************* config
    SEED=57
    
    np.random.seed(SEED)
    
    random.seed(SEED) # now works
    
    GRID_SIZE = 6
    HOLE_DENSITY = 7/36 # fixed
    SUCCESS_RATE = 0.7
    GAMMA = 0.99
    EPSILON = 0.1  # For epsilon-greedy target policy
    
    N_EPISODES = 2000  # Number of episodes per run
    N_TRIALS = 5     # Number of independent runs (20+ recommended but VERY HEAVYYYY)
    # Alpha values to test
    ALPHAS = [0.01, 0.1, 0.5, 0.9, 0.99]
    #ALPHAS=[0.5] # first test
    
    
    # print to test vars
    
    print(f"\nConfiguration:")
    print(f"  Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Hole Density: {HOLE_DENSITY*100}%")
    print(f"  Success Rate: {SUCCESS_RATE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Epsilon: {EPSILON}")
    print(f"  Episodes per trial: {N_EPISODES}")
    print(f"  Number of trials: {N_TRIALS}")
    
    # ok lets starttttt
    
    print("\n" + "="*70)
    print("STEP 1: Creating Environment")
    print("="*70)
    
    env = FrozenLakeEnvironment(
        size=GRID_SIZE,
        success_rate=SUCCESS_RATE,
        seed=SEED
    )
    
    print(f"\nEnvironment Map:")
    print(env.get_map_string())
    print(f"\nStates: {env.n_states}, Actions: {env.n_actions}")
    
    
    
    # ****************** GRAPH 1: ALPHA SWEEP (ON-POLICY)
    print("\n" + "="*70)
    print("STEP 2: Running Graph 1 - Alpha Sweep for On-Policy MC")
    print("="*70)
    print(f"Testing α values: {ALPHAS}")
    
    graph1_results = plot_graph1_alpha_sweep(
        env=env,
        alphas=ALPHAS,
        epsilon=EPSILON,
        n_episodes=N_EPISODES,
        n_trials=N_TRIALS,
        v_star_start=None  # We don't have DP baseline....
    )
    
    # Find best alpha
    best_alpha = None
    best_final_v = -np.inf
    
    print("\nGraph 1 Results:")
    for alpha, results in graph1_results.items():
        final_v_mean = np.mean(results[:, -100:])  # Average last 100 episodes
        final_v_std = np.std(results[:, -100:])
        print(f"  α = {alpha:5.2f}: V(s_start) = {final_v_mean:.4f} ± {final_v_std:.4f}")
        
        if final_v_mean > best_final_v:
            best_final_v = final_v_mean
            best_alpha = alpha
    
    print(f"\nBest α for on-policy MC: {best_alpha} (Final V = {best_final_v:.4f})")
    
    # ******************** GRAPH 3: BEHAVIOUR POLICIES WITH α=1
    print("\n" + "="*70)
    print("STEP 3: Running Graph 3 - Behaviour Policies with α=1")
    print("="*70)
    
    graph3_results = plot_graph3_behaviour_policies_alpha1(
        env=env,
        epsilon=EPSILON,
        n_episodes=N_EPISODES,
        n_trials=N_TRIALS,
        v_star_start=None
    )
    
    print("\nGraph 3 Results (α=1):")
    for policy_name, results in graph3_results.items():
        final_v_mean = np.mean(results[:, -100:])
        final_v_std = np.std(results[:, -100:])
        print(f"  {policy_name:20s}: V(s_start) = {final_v_mean:.4f} ± {final_v_std:.4f}")
    
    # ************************** GRAPH 2: BEHAVIOUR POLICIES WITH BEST α 
    print("\n" + "="*70)
    print("STEP 4: Running Graph 2 - Behaviour Policies with Best α (Optional)")
    print("="*70)
    print(f"Using best α = {best_alpha}")
    
    graph2_results = plot_graph2_behaviour_policies_best_alpha(
        env=env,
        best_alpha=best_alpha,
        epsilon=EPSILON,
        n_episodes=N_EPISODES,
        n_trials=N_TRIALS,
        v_star_start=None
    )
    
    print(f"\nGraph 2 Results (α={best_alpha}):")
    for policy_name, results in graph2_results.items():
        final_v_mean = np.mean(results[:, -100:])
        final_v_std = np.std(results[:, -100:])
        print(f"  {policy_name:20s}: V(s_start) = {final_v_mean:.4f} ± {final_v_std:.4f}")
    
    # *******************************   GRAPH 4: BEST COMPARISON
    print("\n" + "="*70)
    print("STEP 5: Creating Graph 4 - Best On-Policy vs Best Off-Policy")
    print("="*70)
    
    plot_graph4_best_comparison(
        graph1_results=graph1_results,
        graph3_results=graph3_results,
        graph2_results=graph2_results,
        best_alpha=best_alpha,
        n_episodes=N_EPISODES,
        n_trials=N_TRIALS,
        v_star_start=None
    )
    
    # *************************************VISUALIZE FINAL POLICY 
    print("\n" + "="*70)
    print("STEP 6: Visualizing Learned Policy")
    print("="*70)
    
    # Train one final agent with best settings to visualize ( FROM THE BEST A GRAPH 3)
    print(f"Training final agent with best α = {best_alpha}...")
    
    # Find best off-policy
    best_offpolicy_name = None
    best_offpolicy_v = -np.inf
    
    for policy_name, results in graph3_results.items():
        final_v = np.mean(results[:, -100:])
        if final_v > best_offpolicy_v:
            best_offpolicy_v = final_v
            best_offpolicy_name = policy_name
    
    policy_map = {
        'Random': RandomPolicy,
        'Scaled': ScaledPolicy,
        'Adaptive': AdaptivePolicy,
        'Recency-Buffered': RecencyBufferedPolicy,
        'Distance-Aware': DistanceAwarePolicy
    }
    
    best_policy_class = policy_map[best_offpolicy_name]
    
    best_policy = best_policy_class(
        env.n_states, env.n_actions, EPSILON
    )

    # for simplicity and consistency with Graph 3, we train the best policy with α=1
    best_policy_alpha = 1.0
    Q_final, _ = train_mc_agent(env, best_policy, N_EPISODES, best_policy_alpha, EPSILON, GAMMA)
    
    # Create visualizations
    print("Creating policy visualization...")
    visualize_policy(Q_final, env, "Learned Policy")
    
    print("Creating value function heatmap...")
    create_value_heatmap(Q_final, env, "Value Function V(s)")
    
    # **************FINAL SUMMARY + GENERAATED FILES
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  1. environment_map.txt - The grid world map")
    print("  2. graph1_alpha_sweep.png - Alpha comparison for on-policy MC")
    print("  3. graph2_behaviour_policies_alpha{best_alpha}.png - Behaviour policies with best α")
    print("  4. graph3_behaviour_policies_alpha1.png - Behaviour policies with α=1")
    print("  5. graph4_best_comparison.png - Best algorithms comparison")
    print("  6. learned_policy.png - Visualization of learned policy")
    print("  7. value_function_v(s)_heatmap.png - Value function heatmap")
    
    print("\nKey Findings:")
    print(f"  - Best α for on-policy: {best_alpha}")
    print(f"  - Best final V(s_start): {best_final_v:.4f}")
    
    # Find best off-policy
    best_offpolicy_name = None
    best_offpolicy_v = -np.inf
    
    for policy_name, results in graph3_results.items():
        final_v = np.mean(results[:, -100:])
        if final_v > best_offpolicy_v:
            best_offpolicy_v = final_v
            best_offpolicy_name = policy_name
    
    print(f"  - Best behaviour policy (α=1): {best_offpolicy_name}")
    print(f"  - Best off-policy V(s_start): {best_offpolicy_v:.4f}")
    
    print("\n" + "="*70)
    
    return {
        'env': env,
        'graph1_results': graph1_results,
        'graph2_results': graph2_results,
        'graph3_results': graph3_results,
        'best_alpha': best_alpha,
        'Q_final': Q_final
    }


if __name__ == "__main__":
    results = main()
    print("\nAll experiments completed successfully!")
    print("Check the generated PNG files for visualizations.")