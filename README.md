# Topological Early-Warning Indicator: Detecting Representation Fracturing Before Validation Loss Degradation

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TDA](https://img.shields.io/badge/TDA-Ripser-blue)](https://github.com/scikit-tda/ripser)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides an empirical testbed and implementation pipeline demonstrating that **Topological Data Analysis (TDA)** metrics—specifically **Persistent Entropy $E(D)$** and Betti numbers ($b_k$)—serve as leading indicators for neural network overfitting and representation degradation.

---

## 📌 Theoretical Hypothesis

* **Core Hypothesis:** As a deep neural network begins to overfit, the geometric manifold structure of intermediate feature activations "fractures" (clusters fragment into noisy, isolated topologies and topological entropy shifts sharply) **before** the scalar cross-entropy validation loss begins its upward degradation inflection.
* **Independent Variable:** Training epochs under an intentionally overfitted setup (subsampled training distribution).
* **Dependent Variables:**
  1. **Validation Loss:** $\mathcal{L}_{\text{val}}(t)$
  2. **Persistent Entropy:** $E(D_t)$ of penultimate layer representation activations.

---

## 🔬 Experimental Setup

To produce rapid, high-signal verification without heavy compute requirements:

* **Dataset:** CIFAR-10 (subsampled to accelerate overfitting).
* **Architecture:** Convolutional Neural Network (CNN) / ResNet backbone.
* **Probing Layer:** Penultimate layer (pre-classification latent embedding space $Z \in \mathbb{R}^d$).
* **Core Libraries:** `torch`, `torchvision`, `ripser`, `numpy`, `matplotlib`.

---

## 📐 Mathematical Formulation & Pipeline

```
  Input Images ──► [ CNN Backbone ] ──► Penultimate Latents Z (N x d)
                                                  │
                                                  ▼
                                       [ Vietoris-Rips Complex ]
                                                  │
                                                  ▼
                                     Persistence Diagram D (H0, H1)
                                                  │
                                                  ▼
                                       Persistent Entropy E(D)
```

### 1. Subsample Latent Activations
Because Vietoris–Rips persistent homology scales cubically with point count ($\mathcal{O}(N^3)$), we sample a fixed evaluation batch of $N = 300\text{–}500$ points across intervals:
1. Pass evaluation inputs through the model.
2. Extract penultimate latent activations $Z = [z_1, z_2, \dots, z_N] \in \mathbb{R}^{N \times d}$.
3. Standardize/normalize feature activations across dimensions.

### 2. Vietoris–Rips Persistent Homology
We construct persistence diagrams $D = \{(b_i, d_i)\}_{i=1}^m$ across dimensions $H_0$ (connected components) and $H_1$ (1D topological loops):
* **Birth time ($b_i$):** Distance threshold where a topological feature appears.
* **Death time ($d_i$):** Distance threshold where a topological feature closes or merges.
* **Persistence / Lifespan:** $L_i = d_i - b_i$.

### 3. Persistent Entropy Computation
From the barcode lifespans, we compute normalized persistence probabilities:

$$p_i = \frac{L_i}{\sum_{j=1}^m L_j}$$

The Persistent Entropy $E(D)$ is calculated as:

$$E(D) = -\sum_{i=1}^m p_i \log_2(p_i)$$

*(Alternatively, track long-lived $H_0$ component counts at fixed filtration scales to quantify representation fragmentation).*

---

## 📊 Telemetry & Validation Tracking

At every epoch, the training loop tracks:
* **Optimization Metrics:** `train_loss`, `val_loss`, `val_acc`
* **Topological Metrics:** `persistent_entropy_H0`, `persistent_entropy_H1`

### Empirical Lead-Time Signature

```
Loss / Entropy
  │
  │                         H0 Persistent Entropy E(D) [Lead Indicator]
  │                               ┌─────────────────────────────
  │                              ╱
  │                             ╱
  │                            ╱
  │    Validation Loss        │              ▲ Upward Inflection
  │    ───────────────────────┼──────────────┼ (Lagging Indicator)
  │                           │              │
  0───────────────────────── t_lead ──────── t_loss ─────────────► Epochs
                              ▲
                 Topological Anomaly Spike:
                 Representation Manifold Fractures
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. Dependencies (`requirements.txt`)
```text
torch
torchvision
numpy
ripser
matplotlib
```

### 3. Run Experiment
```bash
python run_topological_overfitting_exp.py
```

The script automatically caches the dataset, computes persistent homology per evaluation step, logs metrics, and exports the dual-axis verification plot (`topological_lead_indicator_compact.png`).

---

## 📁 Repository Structure

```
├── run_topological_overfitting_exp.py  # End-to-end training and TDA pipeline
├── topological_lead_indicator_compact.png      # Output dual-axis lead-time figure
├── requirements.txt                    # Python package dependencies
├── README.md                           # Methodology and pipeline documentation
└── .gitignore                          # Standard ignore rules
```

---

## 📜 License
This project is licensed under the MIT License.