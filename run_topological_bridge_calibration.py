import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
import persim

# =============================================================================
# 1. Target Invariant Manifold & Phase-Space Topology Setup
# =============================================================================
np.random.seed(42)
total_steps = 500
t_axis = np.arange(1, total_steps + 1)
omega_target = 0.12  # Canonical angular velocity

# Canonical Target Limit Cycle in Tangent Bundle T(M)
theta_target = np.linspace(0, 2 * np.pi, 250)
tgt_x = np.cos(theta_target)
tgt_y = np.sin(theta_target)
tgt_vx = -omega_target * np.sin(theta_target)
tgt_vy =  omega_target * np.cos(theta_target)
tgt_speed = np.sqrt(tgt_vx**2 + tgt_vy**2)
tgt_ax = -omega_target**2 * np.cos(theta_target)
tgt_ay = -omega_target**2 * np.sin(theta_target)
tgt_curv = np.abs(tgt_vx * tgt_ay - tgt_vy * tgt_ax) / (tgt_speed**3)

# 5D Tangent-Bundle Reference: [x, y, vx, vy, kappa]
target_bundle = np.column_stack([tgt_x, tgt_y, 2.0 * tgt_vx, 2.0 * tgt_vy, 0.3 * tgt_curv])
# Fixed global scaling reference (prevents scale-invariance erasure)
ref_scale = np.std(target_bundle, axis=0) + 1e-8
target_bundle_norm = target_bundle / ref_scale
target_dgm = ripser(target_bundle_norm, maxdim=1)['dgms'][1]

# =============================================================================
# 2. Simulate Realistic Kinematics with Drift & Topological Pullback
# =============================================================================
r = np.zeros(total_steps)
theta = np.zeros(total_steps)
r[0] = 0.40
theta[0] = 0.0

for t in range(1, total_steps):
    theta[t] = theta[t - 1] + omega_target + np.random.normal(0, 0.005)
    
    if t < 100:
        # Phase 1: Supervised convergence
        dr = 0.025 * (1.0 - r[t - 1]) + np.random.normal(0, 0.012)
    elif 100 <= t < 350:
        # Phase 2: Autonomous run under external drift with homological restorative force
        drift = 0.038 * np.sin(t / 28.0)
        pullback = 0.032 * (1.0 - r[t - 1])
        dr = pullback + drift + np.random.normal(0, 0.015)
    else:
        # Phase 3: Audited convergence under restored supervision
        dr = 0.030 * (1.0 - r[t - 1]) + np.random.normal(0, 0.010)
        
    r[t] = np.clip(r[t - 1] + dr, 0.1, 1.8)

# Coordinates and differential kinematics
traj_x = r * np.cos(theta)
traj_y = r * np.sin(theta)
vx = np.gradient(traj_x)
vy = np.gradient(traj_y)
speed = np.sqrt(vx**2 + vy**2) + 1e-8
ax = np.gradient(vx)
ay = np.gradient(vy)
curvature = np.abs(vx * ay - vy * ax) / (speed**3)

traj_bundle = np.column_stack([traj_x, traj_y, 2.0 * vx, 2.0 * vy, 0.3 * curvature])

# Ground Truth Multi-Component Scalar Reward (Target: 10.0)
r_err = np.abs(r - 1.0)
speed_err = np.abs(speed - omega_target)
curv_err = np.abs(curvature - 1.0)

r_radial = 6.0 * np.exp(-4.5 * (r_err ** 2))
r_tangent = 2.5 * np.exp(-14.0 * (speed_err ** 2))
r_curv = 1.5 * np.exp(-2.0 * (curv_err ** 2))
ground_truth_scalar = np.clip(r_radial + r_tangent + r_curv + np.random.normal(0, 0.08, total_steps), 0.0, 10.0)

# =============================================================================
# 3. Metric-Preserving Persistent Homology Filtration
# =============================================================================
raw_w_dists = []
window_size = 45

for i in range(1, total_steps + 1):
    current_pts = traj_bundle[max(0, i - window_size):i]
    
    if len(current_pts) < 18:
        raw_w_dists.append(3.0)
        continue
    
    # Normalize by fixed reference frame (retains size and shape distortions)
    b_norm = current_pts / ref_scale
    res = ripser(b_norm, maxdim=1)
    h1_dgm = res['dgms'][1]
    
    if len(h1_dgm) == 0:
        raw_w_dists.append(3.0)
    else:
        w_dist = persim.wasserstein(h1_dgm, target_dgm, matching=False)
        raw_w_dists.append(float(w_dist))

raw_w_dists = np.array(raw_w_dists)

# Smooth topological distance to eliminate boundary jumping
smooth_w = np.convolve(raw_w_dists, np.ones(6) / 6.0, mode='same')
phi_global = np.exp(-1.6 * smooth_w)

# =============================================================================
# 4. Calibration & Invariant Bridge Fusion
# =============================================================================
# Local kinematic regularization
psi_local = -0.8 * np.abs(np.diff(speed, prepend=speed[0]))

# Dynamic linear mapping calibrated from Phase 1
p1_phi = phi_global[:100]
p1_scalar = ground_truth_scalar[:100]
slope, intercept = np.polyfit(p1_phi, p1_scalar, 1)

# Calibrated Invariant Pseudo-Reward
calibrated_pseudo_reward = np.clip(slope * phi_global + intercept + psi_local, 0.0, 10.0)

# =============================================================================
# 5. Assemble Active Stream & Verification Metrics
# =============================================================================
active_reward = np.zeros(total_steps)
active_reward[:100] = ground_truth_scalar[:100]
active_reward[100:350] = calibrated_pseudo_reward[100:350]
active_reward[350:] = ground_truth_scalar[350:]

phase2_mae = np.mean(np.abs(calibrated_pseudo_reward[100:350] - ground_truth_scalar[100:350]))
gap_100 = np.abs(active_reward[100] - active_reward[99])
gap_350 = np.abs(active_reward[350] - active_reward[349])

print(f"\n=======================================================")
print(f"METRIC-PRESERVING TOPOLOGICAL BRIDGE RESULTS")
print(f"=======================================================")
print(f"Phase 2 Mean Absolute Alignment Gap: {phase2_mae:.4f} reward units")
print(f"Handover In Discontinuity (t=100)  : {gap_100:.4f}")
print(f"Handover Out Discontinuity (t=350) : {gap_350:.4f}")
print(f"=======================================================\n")

# =============================================================================
# 6. Presentation Visualization
# =============================================================================
plt.close('all')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.0), dpi=200, sharex=True)

c_scalar = '#e74c3c'  # Red
c_topo = '#00acc1'    # Cyan
c_active = '#263238'  # Dark Slate

# --- Subplot 1: Active Reward Stream ---
l_gt, = ax1.plot(t_axis, ground_truth_scalar, color=c_scalar, linestyle=':', alpha=0.6, label='Scalar Ground Truth $R_{\mathrm{env}}$')
l_ps, = ax1.plot(t_axis, calibrated_pseudo_reward, color=c_topo, linestyle='--', alpha=0.85, label='Fiber-Bundle Invariant Reward $\mathcal{R}_{\mathrm{bridge}}$')
l_act, = ax1.plot(t_axis, active_reward, color=c_active, linewidth=2.2, label='Active Training Signal')

ax1.axvspan(0, 100, color='#eeeeee', alpha=0.5)
ax1.axvspan(100, 350, color='#e0f7fa', alpha=0.6)
ax1.axvspan(350, 500, color='#ffebee', alpha=0.5)

ax1.text(50, 0.8, 'Phase 1: Co-Calibration', ha='center', fontsize=8.5, fontweight='bold', color='#424242')
ax1.text(225, 0.8, 'Phase 2: Topological Blackout Control', ha='center', fontsize=8.5, fontweight='bold', color='#006064')
ax1.text(425, 0.8, 'Phase 3: Audit', ha='center', fontsize=8.5, fontweight='bold', color='#b71c1c')

ax1.set_ylabel('Reward Amplitude', fontsize=10, fontweight='bold')
ax1.set_ylim(-0.5, 11.5)
ax1.legend(handles=[l_act, l_ps, l_gt], loc='upper left', framealpha=0.9, fontsize=8.5, ncol=3)
ax1.set_title('Metric-Preserving Topological Bridge: Co-Tracking Under Environmental Drift', fontsize=11, fontweight='bold', pad=10)

# --- Subplot 2: Disparity Residuals ---
residuals = np.abs(calibrated_pseudo_reward - ground_truth_scalar)
ax2.plot(t_axis, residuals, color='#c62828', linewidth=1.4, label='Disparity $|\mathcal{R}_{\mathrm{bridge}} - R_{\mathrm{env}}|$')
ax2.axhline(phase2_mae, color='#c62828', linestyle='--', linewidth=1.2, label=f'Phase 2 Mean Gap ({phase2_mae:.2f})')

ax2.axvspan(0, 100, color='#eeeeee', alpha=0.5)
ax2.axvspan(100, 350, color='#e0f7fa', alpha=0.6)
ax2.axvspan(350, 500, color='#ffebee', alpha=0.5)

ax2.set_xlabel('Environment Step ($t$)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Disparity Error', fontsize=10, fontweight='bold')
ax2.set_ylim(0, 2.5)
ax2.set_xlim(0, 500)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=8.5)

plt.tight_layout()
plt.savefig('topological_handover_calibration.png', bbox_inches='tight', dpi=300)
print("[Done] Calibration figure saved as 'topological_handover_calibration.png'")
plt.show()