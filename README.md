# Topological Bridge: Invariant Reinforcement Learning Under Verification Blackouts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Ripser TDA](https://img.shields.io/badge/TDA-Ripser%20%26%20Persim-orange.svg)](https://ripser.scikit-tda.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An autonomous reinforcement learning framework where **Tangent-Bundle Persistent Homology** ($H_1$) serves as an intrinsic, coordinate-invariant pseudo-reward oracle. 

This framework establishes a continuous bridge between point-wise scalar feedback and global manifold topology, enabling policy guidance and structural containment across unobserved blackout environments.

---

## 1. Problem Statement & Motivation

In standard Reinforcement Learning (RL), agents rely on scalar reward signals $R(s_t, a_t)$ calibrated against explicit coordinate frames. In sparse or GPS-denied domains, supervisory feedback can be severed (**Verification Blackout**).

Under scalar reward starvation, policies encounter three failure modes:
* **Gradient Starvation:** $\nabla_\theta J(\pi_\theta) \to 0$, causing drift into unrecoverable state space.
* **Coordinate Brittleness:** Scalar functions penalize harmless phase shifts or high-frequency sensor noise.
* **Dead-Reckoning Drift:** Cumulative integration errors during open-loop execution.

This framework introduces a **Topological Handover Protocol** that derives surrogate guidance directly from the **algebraic topology of trajectory flow in phase space**.

---

## 2. Theoretical Architecture

Trajectory Window τ = { s_(t-k), ..., s_t }
                                  │
           ┌──────────────────────┴──────────────────────┐
           ▼                                             ▼
 Base Manifold Invariant                      Tangent Fiber Bundle
(Vietoris-Rips Filtration)                    (Differential Flow)
           │                                             │
  H₁ Persistence Diagram D_τ                  Velocity / Curvature
           │                                             │
Wasserstein Matching W₁(D_τ, D*)               Flow Alignment ⟨v, v*⟩
           │                                             │
           ▼                                             ▼
   Macro Structural Score                      Micro Kinematic Penalty
        Φ_global(D_τ)                               Ψ_local(v, κ)
           └──────────────────────┬──────────────────────┘
                                  ▼
                    Hodge-Decomposed Bridge Reward
                    R_bridge = α·Φ_global + β·Ψ_local

                    ### A. Tangent-Bundle Lifting ($T\mathcal{M}$)
To account for traversal direction, speed, and acceleration, trajectory states are lifted to the **5D Tangent Bundle**:

$$
\mathbf{z}(t) = \Big( x(t), \; y(t), \; \gamma_v \dot{x}(t), \; \gamma_v \dot{y}(t), \; \gamma_\kappa \kappa(t) \Big) \in \mathbb{R}^5
$$

where instantaneous planar curvature $\kappa(t)$ is defined as:

$$
\kappa(t) = \frac{|\dot{x}\ddot{y} - \dot{y}\ddot{x}|}{\left(\dot{x}^2 + \dot{y}^2\right)^{3/2}}
$$

---

### B. Metric-Preserving Persistent Homology
Vietoris-Rips complexes $\mathcal{VR}_\epsilon(\mathbf{z}_\tau)$ are computed across sliding windows $\tau$. The dominant 1-cycle ($H_1$) characterizes loop stability:
* **Birth ($b$):** Spatial scale where points connect into an enclosed cycle.
* **Death ($d$):** Scale where the cycle is filled by simplices.
* **Lifespan ($L = d - b$):** Proportional to orbit diameter and boundary integrity.

---

### C. Hodge-Style Surrogate Bridge
The unified surrogate reward decomposes into global topology and local kinematic regularizers:

$$
\mathcal{R}_{\text{bridge}}(t) = \text{AffineFit}\left( \frac{\max_i(d_i - b_i)}{1.0 + \lambda \cdot \mathcal{W}_1(D_\tau, D^*)} \right) + \Psi_{\text{local}}(\dot{s}, \ddot{s})
$$

---

## 3. Experimental Protocol: 3-Phase Handover

The system is evaluated over a 500-step dynamic trajectory with external perturbations:

| Phase | Steps ($t$) | Supervision Mode | Objective |
| :--- | :--- | :--- | :--- |
| **Phase 1: Co-Calibration** | $1 \le t \le 100$ | Full Ground Truth ($R_{\text{env}}$) | Calibrate affine mapping from $H_1$ persistence to scalar reward scale. |
| **Phase 2: Blackout Control** | $101 \le t \le 350$ | **Autonomous Invariant ($r_{\text{topo}}$)** | External drift $F_{\text{drift}} = A \sin(\omega t)$ applied; policy guided by topological pullback. |
| **Phase 3: Audit** | $351 \le t \le 500$ | Full Ground Truth ($R_{\text{env}}$) | Restores ground truth to measure trajectory fidelity and drift bounds. |

---

## 4. Evaluation Metrics

* **Phase 2 Mean Absolute Alignment Gap:**
  $$
  \text{MAE}_{\text{P2}} = \frac{1}{250} \sum_{t=101}^{350} \big| r_{\text{topo}}(t) - R_{\text{env}}(t) \big|
  $$

* **Boundary Handover Discontinuity:**
  $$
  \Delta R_{\text{in}} = |r(100) - r(99)|, \quad \Delta R_{\text{out}} = |r(350) - r(349)|
  $$

---

## 5. Repository Structure

├── run_topological_bridge_calibration.py  # Core simulation and TDA engine
├── topological_handover_calibration.png   # Output telemetry plots
├── requirements.txt                      # Project dependencies
└── README.md                             # Architecture documentation

## 6. Installation & Quickstart

```bash
# Clone the repository
git clone [https://github.com/YourUsername/topological-invariant-rl.git](https://github.com/YourUsername/topological-invariant-rl.git)
cd topological-invariant-rl

# Install dependencies
pip install numpy matplotlib ripser persim scipy

# Run calibration benchmark
python run_topological_bridge_calibration.py

7. Results & Telemetry

The calibration run outputs a dual-panel verification figure:Active Reward Stream (Top): Demonstrates seamless transitions at $t=100$ and $t=350$ with zero policy shock.Disparity Residuals (Bottom): Tracks $|r_{\text{topo}} - R_{\text{env}}|$, confirming that the topological surrogate dynamically responds to environmental disturbances.8. LicenseDistributed under the MIT License. See LICENSE for details.