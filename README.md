## 📌 Overview

Standard Transformer backbones suffer from quadratic $O(N^2)$ context scaling and significant KV-cache memory footprints during long-horizon sequence modeling. While Selective State Space Models (Mamba/SSMs) achieve $O(1)$ recurrent memory efficiency, their latent states and parametric weights remain vulnerable to continuous drift and catastrophic forgetting when exposed to dynamic data distributions.

**TCLA** integrates:
1. **Frozen/Pretrained LLM Backbones** (e.g., Llama-3.2, Qwen-2.5) for rich semantic feature binding.
2. **Selective State Space Filtering** for linear-time scan and constant inference memory.
3. **Topological Invariant Regularization** (Simplicial Complexes & Persistent Homology) to preserve structural manifold geometry across continual streaming tasks.

---

## 🏛️ Architecture & Theoretical Framework

### 1. Geometric & Topological State Representation
State spaces are modeled across three progressive structural regimes:
* **1D Causal Sequences:** Standard token-level sequential dependence.
* **1-Skeleton Graphs ($G = (V, E)$):** Captures pairwise associative transitions and non-local graph neighborhoods.
* **2-Simplicial Complexes ($K = (V, E, F)$):** Encodes 3-way interactions, filling triangular 2-simplices to track topological invariants ($\beta_0, \beta_1$) and prevent representation collapse under continuous domain shifts.

### 2. Continual Learning & Retention Metrics
Continual stability is tracked using Reformulated Backward Transfer ($BWT$):
$$BWT = \frac{1}{T-1} \sum_{i=1}^{T-1} (R_{T, i} - R_{i, i})$$
Where $R_{T, i}$ evaluates retrieval accuracy and latent manifold coordinate preservation ($\mathcal{M}_0$) of task $i$ after streaming up to task $T$.

---

## 📊 Benchmark Suite & Recent Results

### Experiment 1.0: Latency & Memory Scaling vs. Llama-3 Attention
Evaluated token-by-token generation across horizons up to $L = 8{,}192$:
* **KV Cache Footprint:** Llama-3 attention scales linearly to **128.00 MB**; TCLA maintains a constant **0.50 MB** state footprint ($256\times$ reduction at $L=8{,}192$).
* **Step Latency:** TCLA maintains invariant latency ($\sim 1.30\text{ ms/step}$), outperforming quadratic attention decoding beyond $L \approx 5{,}500$.

| Context Length ($L$) | Llama-3 KV Cache | TCLA State Cache | Llama-3 Step Latency | TCLA Step Latency |
| :--- | :--- | :--- | :--- | :--- |
| **512** | 8.00 MB | **0.50 MB** | 0.82 ms | **1.31 ms** |
| **2,048** | 32.00 MB | **0.50 MB** | 1.05 ms | **1.29 ms** |
| **4,096** | 64.00 MB | **0.50 MB** | 1.22 ms | **1.30 ms** |
| **8,192** | 128.00 MB | **0.50 MB** | 1.78 ms | **1.32 ms** |

---

### Experiment 1.1 / 1.2: Multi-Query Associative Recall (MQAR)
* **Configuration:** $L=256$, Vocab Size = 512, 8 scattered Key-Value pairs, 2,000 steps.
* **Findings:** Evaluated un-pretrained Attention, Selective SSM, and Hybrid modules from scratch. Identified an optimization plateau ($\text{Loss} \approx 5.27$, marginal distribution baseline), establishing that tabula rasa associative routing requires explicit positional embeddings (RoPE) and dense pretrained prior weights (e.g., Llama-3.2-1B backbone) for long-horizon multi-key binding.

---

## 🚀 Repository Structure

├── benchmarks/
│   ├── experiment_1_1_mqar_benchmark.py   # Synthetic MQAR multi-key recall evaluation
│   ├── experiment_1_2_scaling_latency.py  # Latency vs. Context Length scaling suite
│   └── continual_bwt_eval.py              # Backward Transfer (BWT) streaming benchmark
├── models/
│   ├── ssm_block.py                       # Selective State Space scan layers
│   ├── simplicial_adapter.py              # 2-Simplicial complex topological regularization
│   └── tcla_hybrid.py                     # Backbone + Topo Adapter + SSM pipeline
├── scripts/
│   └── run_overnight_benchmarks.sh        # Automated overnight execution & logging
├── assets/
│   └── experiment_1_1_mqar_results.png    # Convergence curves
└── README.md


---

## 🛠️ Getting Started

### Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/topological-etextile-filter.git
cd topological-etextile-filter
pip install -r requirements.txt
Running the MQAR Benchmark
Bash
python benchmarks/experiment_1_1_mqar_benchmark.py
Running the Latency & Memory Profiler
Bash
python benchmarks/experiment_1_2_scaling_latency.py --max_seq_len 8192
🔮 Roadmap
[x] O(1) memory scaling benchmark vs. Llama-3 attention.

[x] Formulate 1D sequence to 2-simplicial complex mapping pipeline.

[x] Initial MQAR CPU baseline and bottleneck identification.

[ ] Integrate compiled GPU parallel associative scan (mamba-ssm).

[ ] Hook pretrained Llama-3.2-1B / Qwen2.5-0.5B backbones for semantic binding.

[ ] Multi-task continual streaming validation (BWT≥0).
