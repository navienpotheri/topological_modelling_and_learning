import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import wasserstein

def generate_synthetic_multimodal_data(n_samples=2000):
    """
    Generates a 2-channel non-stationary biometric/kinematic proxy signal:
    - State A (t = 0 to 10s): 1.0 Hz fundamental oscillation
    - State B (t = 10 to 20s): 2.5 Hz oscillation with 1.8x amplitude expansion
    - Contaminated with high-frequency micro-vibrations and random burst contact artifacts.
    """
    np.random.seed(42)
    time = np.linspace(0, 20, n_samples)

    # Underlying structural state manifold
    s1 = np.where(time < 10, np.sin(2 * np.pi * 1.0 * time), 1.8 * np.sin(2 * np.pi * 2.5 * time))
    s2 = np.where(time < 10, np.cos(2 * np.pi * 1.0 * time), 1.8 * np.cos(2 * np.pi * 2.5 * time))
    clean_signal = np.stack([s1, s2], axis=1)

    # Micro-artifact noise: Gaussian white noise + non-stationary contact burst spikes
    micro_artifacts = np.random.normal(0, 0.15, clean_signal.shape)
    burst_mask = (np.random.rand(n_samples, 1) > 0.96).astype(float)
    contact_spikes = burst_mask * np.random.normal(0, 1.2, clean_signal.shape)

    noisy_stream = clean_signal + micro_artifacts + contact_spikes
    return time, clean_signal, noisy_stream

def delay_embedding(series, delay=4, dimension=3):
    """Constructs state-space point cloud using Takens' delay embedding."""
    N = len(series) - (dimension - 1) * delay
    embedded = np.empty((N, dimension * series.shape[1]))
    for i in range(dimension):
        embedded[:, i * series.shape[1]:(i + 1) * series.shape[1]] = series[i * delay : i * delay + N]
    return embedded

def compute_windowed_persistence(data_stream, win_size=150, step=25, delay=4, dim=3):
    """Slides a temporal window and computes Vietoris-Rips persistence diagrams."""
    diagrams = []
    timestamps_idx = []
    
    for start in range(0, len(data_stream) - win_size, step):
        window = data_stream[start : start + win_size]
        point_cloud = delay_embedding(window, delay=delay, dimension=dim)
        
        # Compute persistent homology up to H1 (loops/cycles)
        result = ripser(point_cloud, maxdim=1)
        diagrams.append(result['dgms'])
        timestamps_idx.append(start + win_size // 2)
        
    return timestamps_idx, diagrams

def filter_diagram(dgm, delta_th=0.20):
    """Eliminates low-persistence micro-artifact noise near the diagonal."""
    if len(dgm) == 0:
        return np.empty((0, 2))
    lifespans = dgm[:, 1] - dgm[:, 0]
    return dgm[lifespans > delta_th]

def main():
    print("1. Generating synthetic biometric/first-person sensor proxy stream...")
    time, clean_signal, noisy_stream = generate_synthetic_multimodal_data()

    print("2. Running Takens' delay embedding and Vietoris-Rips filtration...")
    timestamps_idx, dgms_history = compute_windowed_persistence(
        noisy_stream, win_size=150, step=25, delay=4, dim=3
    )

    print("3. Computing Wasserstein distances across successive topological states...")
    wasserstein_deltas = []
    delta_th = 0.20

    for i in range(1, len(dgms_history)):
        dgm_prev_h1 = filter_diagram(dgms_history[i - 1][1], delta_th=delta_th)
        dgm_curr_h1 = filter_diagram(dgms_history[i][1], delta_th=delta_th)
        
        # 2-Wasserstein distance between filtered topological signatures
        w_dist = wasserstein(dgm_prev_h1, dgm_curr_h1)
        wasserstein_deltas.append(w_dist)

    # 4. Metric Evaluation: Wasserstein Signal-to-Artifact Ratio (WSAR)
    transition_sample = len(time) // 2
    transition_eval_idx = np.argmin(np.abs(np.array(timestamps_idx[1:]) - transition_sample))

    inter_state_w = np.max(wasserstein_deltas[max(0, transition_eval_idx - 2) : transition_eval_idx + 3])
    intra_state_indices = [k for k in range(len(wasserstein_deltas)) if abs(k - transition_eval_idx) > 2]
    intra_state_w = np.max([wasserstein_deltas[k] for k in intra_state_indices]) if intra_state_indices else 1e-8

    wsar = inter_state_w / (intra_state_w + 1e-8)

    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS")
    print("=" * 50)
    print(f"Inter-State Topological Transition Peak: {inter_state_w:.4f}")
    print(f"Max Intra-State Noise Fluctuation:       {intra_state_w:.4f}")
    print(f"Wasserstein Signal-to-Artifact Ratio:    {wsar:.2f}")
    
    if wsar > 1.0:
        print("\n[VERDICT: PASS] Inter-state structural change is completely separable from micro-artifacts.")
    else:
        print("\n[VERDICT: FAIL] Artifact noise overlaps with structural state transitions.")
    print("=" * 50)

    # 5. Visualization
    time_pts = time[timestamps_idx[1:]]
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    
    axes[0].plot(time, clean_signal[:, 0], 'g', label='Ground Truth Channel 1')
    axes[0].set_ylabel('Clean Amplitude')
    axes[0].set_title('Sensor Ground Truth (Macro Structural Shift at t = 10s)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, noisy_stream[:, 0], 'k', alpha=0.7, label='Noisy Stream (Artifacts + Contact Spikes)')
    axes[1].set_ylabel('Sensor Signal')
    axes[1].set_title('Raw Input with Severe Non-Stationary Micro-Artifacts')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_pts, wasserstein_deltas, 'r-o', lw=2, label='Topological Layer $\\Delta W_2$ Response')
    axes[2].axhline(y=intra_state_w, color='blue', linestyle='--', label=f'Max Noise Floor ({intra_state_w:.2f})')
    axes[2].set_ylabel('Wasserstein Distance')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title(f'Topological State Extraction (WSAR = {wsar:.2f})')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()