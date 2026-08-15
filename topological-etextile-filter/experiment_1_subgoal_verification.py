import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

def generate_trajectories(n_points=300):
    """
    Simulates three long-horizon trajectory rollouts:
    1. Valid Trajectory: Successfully navigates a closed-loop sub-goal (H1 cycle exists).
    2. Hallucinated Shortcut (Collapsed): Cuts through the obstacle/sub-goal (H1 -> 0).
    3. Premature Termination (Incomplete Arc): Stops halfway without enclosing the sub-goal (H1 -> 0).
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    
    # 1. Valid Execution: True spatial topological loop
    valid_x = 2.0 * np.cos(theta) + np.random.normal(0, 0.05, n_points)
    valid_y = 2.0 * np.sin(theta) + np.random.normal(0, 0.05, n_points)
    valid_traj = np.stack([valid_x, valid_y], axis=1)

    # 2. Hallucinated Shortcut: Straight linear collapse
    short_x = np.linspace(-2.0, 2.0, n_points) + np.random.normal(0, 0.05, n_points)
    short_y = np.linspace(-0.2, 0.2, n_points) + np.random.normal(0, 0.05, n_points)
    shortcut_traj = np.stack([short_x, short_y], axis=1)

    # 3. Premature Termination: Half arc (fails closure)
    theta_half = np.linspace(0, np.pi, n_points)
    incomp_x = 2.0 * np.cos(theta_half) + np.random.normal(0, 0.05, n_points)
    incomp_y = 2.0 * np.sin(theta_half) + np.random.normal(0, 0.05, n_points)
    incomplete_traj = np.stack([incomp_x, incomp_y], axis=1)

    return valid_traj, shortcut_traj, incomplete_traj

def evaluate_h1_persistence(trajectory, persistence_threshold=0.80):
    """
    Computes Vietoris-Rips persistent homology and verifies if a 
    dominant H1 topological generator spans the required threshold.
    """
    result = ripser(trajectory, maxdim=1)
    h1_diagram = result['dgms'][1]

    if len(h1_diagram) == 0:
        max_lifetime = 0.0
    else:
        lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
        # Ignore infinite points if any, take maximum finite persistent lifespan
        finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
        max_lifetime = np.max(finite_lifetimes) if len(finite_lifetimes) > 0 else 0.0

    passed = max_lifetime >= persistence_threshold
    return passed, max_lifetime, h1_diagram

def main():
    print("=" * 65)
    print("EXPERIMENT 1: INVARIANT SUB-GOAL VERIFICATION VIA H1 HOMOLOGY")
    print("=" * 65)

    valid_traj, shortcut_traj, incomplete_traj = generate_trajectories()

    trajectories = [
        ("Valid Trajectory (Closed Sub-goal Loop)", valid_traj),
        ("Hallucinated Shortcut (Collapsed)", shortcut_traj),
        ("Premature Termination (Incomplete)", incomplete_traj)
    ]

    persistence_threshold = 0.80
    results = []

    for name, traj in trajectories:
        passed, max_life, dgm = evaluate_h1_persistence(traj, persistence_threshold)
        status = "PASSED (Advance Context)" if passed else "REJECTED (Hallucination Detected)"
        results.append((name, traj, passed, max_life, dgm))
        print(f"\nScenario: {name}")
        print(f"  Max H1 Persistence Lifetime: {max_life:.4f}")
        print(f"  Oracle Decision:             {status}")

    print("\n" + "=" * 65)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, (name, traj, passed, max_life, dgm) in enumerate(results):
        # Row 1: Spatial Trajectory
        axes[0, idx].plot(traj[:, 0], traj[:, 1], 'b.-', alpha=0.6, label='Trajectory')
        axes[0, idx].plot(0, 0, 'rx', markersize=10, mew=2, label='Obstacle/Subgoal Target')
        axes[0, idx].set_title(f"{name}\nResult: {'PASS' if passed else 'FAIL'}", fontsize=10)
        axes[0, idx].set_xlim(-3, 3)
        axes[0, idx].set_ylim(-3, 3)
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].legend(loc='upper right', fontsize=8)

        # Row 2: Persistence Barcode (H1 Lifespans)
        axes[1, idx].set_title(f"H1 Barcode (Max Life = {max_life:.2f})", fontsize=10)
        if len(dgm) > 0:
            for j, (b, d) in enumerate(dgm):
                if np.isfinite(d):
                    axes[1, idx].plot([b, d], [j, j], 'g-', lw=2)
        axes[1, idx].axvline(x=persistence_threshold, color='r', linestyle='--', label=f'Threshold ({persistence_threshold})')
        axes[1, idx].set_xlabel('Filtration Scale ε')
        axes[1, idx].set_ylabel('Generator Index')
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()