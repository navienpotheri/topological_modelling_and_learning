# Invariant Sub-Goal Verification via $H_1$ Persistent Homology

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TDA: Ripser](https://img.shields.io/badge/TDA-Ripser%20%2F%20Persim-orange.svg)](https://ripser.scikit-tda.org/)

A Topological Data Analysis (TDA) runtime oracle that verifies structural task invariants in long-horizon autoregressive robotic manipulation rollouts.

---

## Overview

Autoregressive policies accumulate compounding errors over extended trajectories, frequently hallucinating task completion without executing necessary geometric sub-goals (e.g., cutting through obstacles or prematurely aborting manipulation loops). 

This repository evaluates $H_1$ persistent homology (tracking 1D topological cycles/holes) to gate the autoregressive context window:

$$\text{If } \max_{i} (d_i - b_i) \ge \delta_{\text{threshold}} \implies \text{Advance Context, else Reject / Re-plan}$$

---

## Experimental Benchmark

The experiment evaluates three candidate rollouts against an obstacle/sub-goal loop ($\delta_{\text{threshold}} = 0.80$):

1. **Valid Trajectory (Closed Loop):** Completes the enclosing path around the target.
2. **Hallucinated Shortcut (Collapsed):** Cuts directly across the coordinate space without circling.
3. **Premature Termination (Incomplete Arc):** Stops halfway without closing the loop manifold.

### Results=================================================================
EXPERIMENT 1: INVARIANT SUB-GOAL VERIFICATION VIA H1 HOMOLOGY
Scenario: Valid Trajectory (Closed Sub-goal Loop)
Max H1 Persistence Lifetime: 3.1006
Oracle Decision:             PASSED (Advance Context)

Scenario: Hallucinated Shortcut (Collapsed)
Max H1 Persistence Lifetime: 0.0230
Oracle Decision:             REJECTED (Hallucination Detected)

Scenario: Premature Termination (Incomplete)
Max H1 Persistence Lifetime: 0.0281
Oracle Decision:             REJECTED (Hallucination Detected)

---

## Quickstart

```bash
git clone [https://github.com/navienpotheri/experiment_1_subgoal_verification.py.git](https://github.com/navienpotheri/experiment_1_subgoal_verification.py.git)
cd experiment_1_subgoal_verification.py
pip install -r requirements.txt
python experiment_1_subgoal_verification.py
License
This project is licensed under the MIT License - see the LICENSE file for details.
