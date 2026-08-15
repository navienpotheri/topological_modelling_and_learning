import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

def compute_betti_profile(point_cloud, epsilons):
    """
    Computes Betti numbers beta_0 (connected components) and beta_1 (1D cycles)
    across a predefined filtration scale range (epsilon vector).
    """
    res = ripser(point_cloud, maxdim=1)
    dgm0 = res['dgms'][0]
    dgm1 = res['dgms'][1]

    betti_0_curve = []
    betti_1_curve = []

    for eps in epsilons:
        # beta_0: birth <= eps < death
        b0 = np.sum((dgm0[:, 0] <= eps) & (eps < dgm0[:, 1]))
        # beta_1: birth <= eps < death
        if len(dgm1) > 0:
            b1 = np.sum((dgm1[:, 0] <= eps) & (eps < dgm1[:, 1]))
        else:
            b1 = 0
        betti_0_curve.append(b0)
        betti_1_curve.append(b1)

    return np.array(betti_0_curve), np.array(betti_1_curve)

def generate_topological_phases(n_points=300):
    """
    Generates three topological phase configurations:
    1. Multi-Modal Phase (Two distinct topological clusters/attractors: beta_0 = 2)
    2. Enclosed Loop Phase (Single boundary loop: beta_0 = 1, beta_1 = 1)
    3. Degenerate Phase (Over-clustered/collapsed points: beta_0 = 1, beta_1 = 0)
    """
    # 1. Multi-Modal Sub-goal Attractors (Two clusters)
    c1 = np.random.normal(loc=[-2.0, 0.0], scale=0.3, size=(n_points // 2, 2))
    c2 = np.random.normal(loc=[2.0, 0.0], scale=0.3, size=(n_points // 2, 2))
    multimodal_traj = np.vstack([c1, c2])

    # 2. Enclosed Orbit Phase (Single loop)
    theta = np.linspace(0, 2 * np.pi, n_points)
    loop_x = 2.0 * np.cos(theta) + np.random.normal(0, 0.08, n_points)
    loop_y = 2.0 * np.sin(theta) + np.random.normal(0, 0.08, n_points)
    loop_traj = np.stack([loop_x, loop_y], axis=1)

    # 3. Collapsed Degenerate Cloud
    collapse_traj = np.random.normal(loc=[0.0, 0.0], scale=0.05, size=(n_points, 2))

    return multimodal_traj, loop_traj, collapse_traj

def main():
    print("=" * 75)
    print("EXPERIMENT 4: TOPOLOGICAL PHASE IDENTIFICATION VIA BETTI PROFILES")
    print("=" * 75)

    multi, loop, collapse = generate_topological_phases()
    epsilons = np.linspace(0.01, 3.0, 150)

    scenarios = [
        ("Multi-Modal Sub-Goal Attractors", multi, "Target: beta_0=2 at intermediate eps"),
        ("Enclosed Orbit Trajectory", loop, "Target: beta_1=1 at intermediate eps"),
        ("Degenerate Collapsed Cluster", collapse, "Target: Rapid collapse to beta_0=1, beta_1=0")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for idx, (name, traj, target_desc) in enumerate(scenarios):
        b0_curve, b1_curve = compute_betti_profile(traj, epsilons)

        # Representative intermediate scale check at eps = 1.0
        eval_idx = np.argmin(np.abs(epsilons - 1.0))
        b0_val = b0_curve[eval_idx]
        b1_val = b1_curve[eval_idx]

        print(f"\nScenario: {name}")
        print(f"  Profile Target: {target_desc}")
        print(f"  At Scale eps = 1.0 -> beta_0: {b0_val}, beta_1: {b1_val}")

        # Trajectory Scatter
        axes[0, idx].scatter(traj[:, 0], traj[:, 1], c='teal', s=15, alpha=0.6)
        axes[0, idx].set_title(f"{name}\n({target_desc})", fontsize=10)
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].axis('equal')

        # Betti Curves
        axes[1, idx].plot(epsilons, b0_curve, 'b-', label=r'$\beta_0$ (Connected Components)')
        axes[1, idx].plot(epsilons, b1_curve, 'r-', label=r'$\beta_1$ (1D Cycles)')
        axes[1, idx].axvline(1.0, color='gray', linestyle='--', alpha=0.6, label=r'Eval Scale $\epsilon=1.0$')
        axes[1, idx].set_title(r"Betti Curves ($\beta_0, \beta_1$ vs. $\epsilon$)", fontsize=10)
        axes[1, idx].set_xlabel(r"Filtration Scale $\epsilon$")
        axes[1, idx].set_ylabel("Betti Number Count")
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].legend(fontsize=8)

    print("\n" + "=" * 75)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()