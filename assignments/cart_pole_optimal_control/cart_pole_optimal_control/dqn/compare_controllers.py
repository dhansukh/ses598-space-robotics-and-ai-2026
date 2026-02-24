#!/usr/bin/env python3
"""
LQR vs DQN Comparison Script.

Runs both controllers in the Gymnasium CartPole-v1 environment under
identical earthquake disturbances and generates a side-by-side comparison
plot with performance metrics.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dqn_agent import DQNAgent

# ─── Configuration ───────────────────────────────────────────────────────────
MAX_STEPS = 500
NUM_WAVES = 5
FREQ_RANGE = [0.5, 4.0]
BASE_AMPLITUDE = 15.0
ENV_TIMESTEP = 0.02

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'dqn_cartpole_earthquake.pth')

os.makedirs(IMAGES_DIR, exist_ok=True)


# ─── Earthquake Generator ────────────────────────────────────────────────────
def make_earthquake_generator(seed=42):
    rng = np.random.RandomState(seed)
    frequencies = rng.uniform(FREQ_RANGE[0], FREQ_RANGE[1], NUM_WAVES)
    phase_shifts = rng.uniform(0, 2 * np.pi, NUM_WAVES)

    def generate(time):
        force = 0.0
        for freq, phase in zip(frequencies, phase_shifts):
            amplitude = BASE_AMPLITUDE * rng.uniform(0.8, 1.2)
            force += amplitude * np.sin(2 * np.pi * freq * time + phase)
        force += rng.normal(0, BASE_AMPLITUDE * 0.1)
        return force

    return generate


# ─── LQR Controller (simplified, continuous) ─────────────────────────────────
class LQRController:
    """Simplified LQR controller matching the ROS2 implementation."""

    def __init__(self):
        M, m, L, g = 1.0, 1.0, 1.0, 9.81

        A = np.array([
            [0, 1, 0, 0],
            [0, 0, (m * g) / M, 0],
            [0, 0, 0, 1],
            [0, 0, ((M + m) * g) / (M * L), 0]
        ])
        B = np.array([
            [0],
            [1 / M],
            [0],
            [-1 / (M * L)]
        ])

        Q = np.diag([5.0, 5.0, 20.0, 20.0])
        R = np.array([[0.5]])

        P = linalg.solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P

    def get_action(self, state):
        """Returns continuous force command."""
        x = np.array(state[:4]).reshape(4, 1)
        u = -self.K @ x
        return float(u[0])


# ─── Run Simulation ──────────────────────────────────────────────────────────
def run_episode(controller_type, earthquake_gen, env_seed=42):
    """Run one episode and return time series data."""
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=env_seed)

    if controller_type == 'dqn':
        agent = DQNAgent(5, 2)
        agent.load_model(MODEL_PATH)
    else:
        lqr = LQRController()

    times, cart_pos, pole_ang, ctrl_forces, eq_forces = [], [], [], [], []

    for t in range(MAX_STEPS):
        eq_force = earthquake_gen(t * ENV_TIMESTEP)

        if controller_type == 'dqn':
            state_aug = np.append(state[:4], eq_force)
            action = agent.select_action(state_aug, evaluate=True)
            force = 10.0 if action == 1 else -10.0
        else:
            force = lqr.get_action(state)
            # Map continuous force to discrete CartPole action
            action = 1 if force > 0 else 0

        next_state, _, terminated, truncated, _ = env.step(action)
        # Apply earthquake perturbation to cart position
        next_state[0] += eq_force * ENV_TIMESTEP * 0.01
        done = terminated or truncated

        times.append(t * ENV_TIMESTEP)
        cart_pos.append(next_state[0])
        pole_ang.append(np.degrees(next_state[2]))
        ctrl_forces.append(force)
        eq_forces.append(eq_force)

        state = next_state
        if done:
            break

    env.close()

    return {
        'times': times,
        'cart_pos': cart_pos,
        'pole_ang': pole_ang,
        'ctrl_forces': ctrl_forces,
        'eq_forces': eq_forces,
        'steps': len(times),
        'max_cart_disp': max(map(abs, cart_pos)),
        'max_pole_angle': max(map(abs, pole_ang)),
        'avg_ctrl_effort': np.mean(np.abs(ctrl_forces)),
    }


def main():
    print("Running LQR vs DQN Comparison...")
    print("=" * 60)

    # Use the same earthquake sequence for fair comparison
    eq_gen_lqr = make_earthquake_generator(seed=42)
    eq_gen_dqn = make_earthquake_generator(seed=42)

    lqr_data = run_episode('lqr', eq_gen_lqr, env_seed=42)
    dqn_data = run_episode('dqn', eq_gen_dqn, env_seed=42)

    # ─── Comparison Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('LQR vs DQN Controller Comparison — Cart-Pole with Earthquake',
                 fontsize=14, fontweight='bold')

    # Cart Position
    axes[0, 0].plot(lqr_data['times'], lqr_data['cart_pos'], color='blue', label='LQR')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('LQR: Cart Position')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot(dqn_data['times'], dqn_data['cart_pos'], color='blue', label='DQN')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Position (m)')
    axes[1, 0].set_title('DQN: Cart Position')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Pole Angle
    axes[0, 1].plot(lqr_data['times'], lqr_data['pole_ang'], color='red', label='LQR')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angle (°)')
    axes[0, 1].set_title('LQR: Pole Angle')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 1].plot(dqn_data['times'], dqn_data['pole_ang'], color='red', label='DQN')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angle (°)')
    axes[1, 1].set_title('DQN: Pole Angle')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # Control Force
    axes[0, 2].plot(lqr_data['times'], lqr_data['ctrl_forces'], color='magenta', label='LQR')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Force (N)')
    axes[0, 2].set_title('LQR: Control Force')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    axes[1, 2].plot(dqn_data['times'], dqn_data['ctrl_forces'], color='magenta', label='DQN')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Force (N)')
    axes[1, 2].set_title('DQN: Control Force')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()
    plot_path = os.path.join(IMAGES_DIR, 'dqn_vs_lqr_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")

    # ─── Metrics Table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: LQR vs DQN")
    print("=" * 60)
    print(f"{'Metric':<30} {'LQR':>12} {'DQN':>12}")
    print("-" * 54)
    print(f"{'Steps survived':<30} {lqr_data['steps']:>12d} {dqn_data['steps']:>12d}")
    print(f"{'Max cart displacement (m)':<30} {lqr_data['max_cart_disp']:>12.3f} {dqn_data['max_cart_disp']:>12.3f}")
    print(f"{'Max pole angle (°)':<30} {lqr_data['max_pole_angle']:>12.2f} {dqn_data['max_pole_angle']:>12.2f}")
    print(f"{'Avg control effort (N)':<30} {lqr_data['avg_ctrl_effort']:>12.1f} {dqn_data['avg_ctrl_effort']:>12.1f}")
    print("=" * 60)

    # Qualitative analysis
    print("\nANALYSIS:")
    if lqr_data['steps'] > dqn_data['steps']:
        print("  → LQR survived longer than DQN")
    elif dqn_data['steps'] > lqr_data['steps']:
        print("  → DQN survived longer than LQR")
    else:
        print("  → Both controllers survived the same number of steps")

    if lqr_data['max_pole_angle'] < dqn_data['max_pole_angle']:
        print("  → LQR achieved tighter pole angle control")
    else:
        print("  → DQN achieved tighter pole angle control")

    if lqr_data['avg_ctrl_effort'] < dqn_data['avg_ctrl_effort']:
        print("  → LQR used less control effort (more efficient)")
    else:
        print("  → DQN used less control effort (more efficient)")

    print("\n  LQR uses continuous optimal control with full state feedback,")
    print("  making it naturally better at smooth stabilization. DQN learns")
    print("  a discrete policy from experience, which can generalize to")
    print("  unseen disturbances but provides less precise control.")


if __name__ == '__main__':
    main()
