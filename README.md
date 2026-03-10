# Monte Carlo Reinforcement Learning – Frozen Lake

Final project for a **Reinforcement Learning course**.

This project explores **Monte Carlo control with importance sampling** for solving the Frozen Lake environment.
The goal is to train an agent to navigate a grid world from the start state to the goal while avoiding holes.

---

## Environment

The environment is based on the Frozen Lake environment from **Gymnasium**, with several custom modifications:

* Custom 6x6 grid
* Configurable success probability
* Modified reward structure
* Negative reward for falling into holes

Example map:

```
SFFFFF
FHFHFF
FFFHFF
HFHFFF
FFFHFH
FFFFFG
```

S = Start
F = Frozen tile
H = Hole
G = Goal

---

## Algorithms

The project implements **Off-Policy Monte Carlo Control with Importance Sampling**.

Update rule:

```
Q(S,A) = Q(S,A) + α * (W / C(S,A)) * (G - Q(S,A))
```

Where:

* G = return
* W = importance sampling weight
* C = cumulative weight

---

## Behaviour Policies Tested

Several behaviour policies were implemented and compared:

* Random Policy
* Scaled Policy
* Adaptive Policy
* Recency-Buffered Policy
* Distance-Aware Policy

These policies control how the agent explores the environment.

---

## Experiments

The project includes multiple experiments:

### Graph 1 – Alpha Sweep

Comparison of different learning rates (α) for on-policy Monte Carlo.

### Graph 2 – Behaviour Policies with Best Alpha

Comparison of behaviour policies using the best α value.

### Graph 3 – Behaviour Policies with α = 1

Evaluation of off-policy performance.

### Graph 4 – Best Algorithm Comparison

Comparison between the best on-policy and off-policy configurations.

---

## Visualizations

The project generates:

* Learning curves
* Policy visualizations
* Value function heatmaps

These visualizations help analyze learning behaviour.

---

## How to Run

Run the main experiment file:

```
python main_runner.py
```

This script will:

* Run all experiments
* Train the agents
* Generate graphs
* Save visualizations

---

## Technologies

* Python
* NumPy
* Gymnasium
* Matplotlib
* Seaborn
* Pandas

---

## Project Structure

```
src/
  main_runner.py
  mc_rl_project.py
  mc_experiments.py
```

---


Course project – Reinforcement Learning.
