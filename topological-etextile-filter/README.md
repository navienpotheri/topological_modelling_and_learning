# Invariant Sub-Goal Verification via $H_1$ Persistent Homology

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TDA: Ripser](https://img.shields.io/badge/TDA-Ripser%20%2F%20Persim-orange.svg)](https://ripser.scikit-tda.org/)

A topological guardrail for long-horizon autoregressive robotic manipulation rollouts. This framework uses **$H_1$ persistent homology barcodes** to verify task invariants (geometric loops/cycles) before advancing the policy context window, preventing compounding drift and hallucinated task completion.

---

## The Problem: Autoregressive Hallucination Drift

During long-horizon manipulation, autoregressive models predict local next-step actions without global topological awareness. This causes policies to:
* **Cut through obstacles** (hallucinating success without completing the required orbit/clearance).
* **Terminate early** (stopping mid-arc while reporting task completion).
* **Accumulate compounding coordinate drift** over long rollout steps ($T > 100$).

---

## Topological Oracle Mechanism

1. **State-Space Trajectory Extraction:** Tracks the spatial coordinates of the end-effector rollout $\mathcal{T} = \{x_t\}_{t=1}^T$.
2. **Vietoris–Rips Complex Filtration:** Constructs persistent homology diagrams over the trajectory point cloud.
3. **$H_1$ Invariant Verification:** Extracts the 1D loop generator lifespan:
   $$\text{Max Lifetime } L = \max_{i} (d_i - b_i)$$
4. **Context Gating Decision Rule:**
   $$\text{Decision} = \begin{cases} \text{ADVANCE CONTEXT} & \text{if } L \ge \delta_{\text{threshold}} \\ \text{REJECT / RE-PLAN} & \text{if } L < \delta_{\text{threshold}} \end{cases}$$

---

## Experimental Benchmark & Results

Three rollout candidates were evaluated against an obstacle avoidance / sub-goal orbit task ($\delta_{\text{threshold}} = 0.80$):

| Scenario | Max $H_1$ Lifetime ($L$) | Oracle Verdict |
| :--- | :--- | :--- |
| **Valid Trajectory (Closed Loop)** | **3.1006** | **PASSED** (Advance Context) |
| **Hallucinated Shortcut (Collapsed)** | **0.0230** | **REJECTED** (Hallucination Detected) |
| **Premature Termination (Incomplete)** | **0.0281** | **REJECTED** (Hallucination Detected) |
