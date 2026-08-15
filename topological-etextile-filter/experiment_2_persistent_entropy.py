cd /c/Users/Navie/topological-etextile-filter

cat << 'EOF' > experiment_2_persistent_entropy.py
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

def calculate_persistent_entropy(diagram):
    """
    Computes Persistent Entropy E(D) for a persistence diagram:
    p_i = (d_i - b_i) / sum(d_k - b_k)
    E(D) = - sum(p_i * log(p_i))
    """
    if len(diagram) == 0:
        return 0.0
    
    lifetimes = diagram[:, 1] - diagram[:, 0]
    finite_lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    
    total_life = np.sum(finite_lifetimes)
    if total_life == 0:
        return 0.0
    
    p = finite_lifetimes / total_life
    entropy = -np.sum(p * np.log(p + 1e-12))
    return entropy

def generate_trajectories(n_points=300):
    t = np.linspace(0, 4 * np.pi, n_points)
    
    # 1. Grounded Policy: Coherent structured attractor
    grounded_x = 2.0 * np.sin(t) + np.random.normal(0, 0.03, n_points)
    grounded_y = 2.0 * np.sin(2 * t) + np.random.normal(0, 0.03, n_points)
    grounded_traj = np.stack([grounded_x, grounded_y], axis=1)

    # 2. Chaotic Hallucination: High-entropy micro-jitter
    chaotic_x = np.cumsum(np.random.normal(0, 0.15, n_points)) + np.random.normal(0, 0.2, n_points)
    chaotic_y = np.cumsum(np.random.normal(0, 0.15, n_points)) + np.random.normal(0, 0.2, n_points)
    chaotic_traj = np.stack([chaotic_x, chaotic_y], axis=1)

    # 3. Mode Collapse: Static collapse with near-zero movement
    collapse_x = np.random.normal(0, 0.01, n_points)
    collapse_y = np.random.normal(0, 0.01, n_points)
    collapse_traj = np.stack([collapse_x, collapse_y], axis=1)

    return grounded_traj, chaotic_traj, collapse_traj

def main():
    print("=" * 70)
    print("EXPERIMENT 2: HALLUCINATION DETECTION VIA PERSISTENT ENTROPY")
    print("=" * 70)

    grounded, chaotic, collapse = generate_trajectories()

    scenarios = [
        ("Grounded Policy Execution", grounded),
        ("Chaotic Hallucination (High Jitter)", chaotic),
        ("Mode Collapse (Degenerate Static)", collapse)
    ]

    # H1 entropy bound for grounded structured execution
    h1_max_threshold = 1.5
    results = []

    for name, traj in scenarios:
        res = ripser(traj, maxdim=1)
        h0_diag = res['dgms'][0]
        h1_diag = res['dgms'][1]

        h0_entropy = calculate_persistent_entropy(h0_diag)
        h1_entropy = calculate_persistent_entropy(h1_diag)

        # Grounded structured trajectories have low H1 entropy (dominant loops)
        passed = (len(h1_diag) > 0) and (h1_entropy <= h1_max_threshold)
        decision = "PASSED (Grounded Policy)" if passed else "REJECTED (Hallucination Detected)"
        
        results.append((name, traj, h0_diag, h1_diag, h1_entropy, passed))
        
        print(f"\nScenario: {name}")
        print(f"  H0 Entropy: {h0_entropy:.4f} | H1 Entropy: {h1_entropy:.4f}")
        print(f"  Entropy Oracle Verdict: {decision}")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
EOF