#!/usr/bin/env python3
"""
DQN Evaluation Script for Cart-Pole with Earthquake Disturbances.

Loads a trained DQN model and evaluates it on the CartPole-v1 environment
with earthquake forces. Generates performance plots and prints metrics
for comparison with the LQR controller.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dqn_agent import DQNAgent

# ─── Configuration ───────────────────────────────────────────────────────────
NUM_EPISODES = 10
MAX_STEPS = 1000

# Earthquake parameters
NUM_WAVES = 5
FREQ_RANGE = [0.5, 4.0]
BASE_AMPLITUDE = 15.0
ENV_TIMESTEP = 0.02

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'dqn_cartpole_earthquake.pth')

os.makedirs(IMAGES_DIR, exist_ok=True)


def make_earthquake_generator():
    frequencies = np.random.uniform(FREQ_RANGE[0], FREQ_RANGE[1], NUM_WAVES)
    phase_shifts = np.random.uniform(0, 2 * np.pi, NUM_WAVES)

    def generate(time):
        force = 0.0
        for freq, phase in zip(frequencies, phase_shifts):
            amplitude = BASE_AMPLITUDE * np.random.uniform(0.8, 1.2)
            force += amplitude * np.sin(2 * np.pi * freq * time + phase)
        force += np.random.normal(0, BASE_AMPLITUDE * 0.1)
        return force

    return generate


def evaluate():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0] + 1
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    agent.load_model(MODEL_PATH)

    all_metrics = []

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        earthquake = make_earthquake_generator()

        cart_positions = []
        pole_angles = []
        control_forces = []
        earthquake_forces = []
        total_reward = 0.0
        time_step = 0

        for t in range(MAX_STEPS):
            eq_force = earthquake(time_step * ENV_TIMESTEP)
            state_augmented = np.append(state[:4], eq_force)

            action = agent.select_action(state_augmented, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Apply earthquake effect on cart position
            next_state[0] += eq_force * ENV_TIMESTEP * 0.01
            done = terminated or truncated

            # Log data
            cart_positions.append(next_state[0])
            pole_angles.append(np.degrees(next_state[2]))
            control_forces.append(10.0 if action == 1 else -10.0)
            earthquake_forces.append(eq_force)

            state = next_state
            time_step += 1
            total_reward += reward

            if done:
                break

        # Compute metrics
        metrics = {
            'episode': episode,
            'steps': time_step,
            'total_reward': total_reward,
            'max_cart_disp': max(map(abs, cart_positions)),
            'max_pole_angle': max(map(abs, pole_angles)),
            'avg_control_effort': np.mean(np.abs(control_forces)),
        }
        all_metrics.append(metrics)

        print(
            f"Episode {episode:2d} | "
            f"Steps: {metrics['steps']:4d} | "
            f"Reward: {metrics['total_reward']:7.1f} | "
            f"Max Cart: {metrics['max_cart_disp']:.3f}m | "
            f"Max Angle: {metrics['max_pole_angle']:.2f}°"
        )

    env.close()

    # ─── Plot Best Episode ────────────────────────────────────────────────
    # Re-run best episode for detailed plotting
    best = max(all_metrics, key=lambda m: m['steps'])
    print(f"\nBest episode: {best['episode']} ({best['steps']} steps)")

    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    earthquake = make_earthquake_generator()

    cart_pos, pole_ang, ctrl_f, eq_f, times = [], [], [], [], []
    for t in range(MAX_STEPS):
        eq_force = earthquake(t * ENV_TIMESTEP)
        state_augmented = np.append(state[:4], eq_force)
        action = agent.select_action(state_augmented, evaluate=True)
        next_state, _, terminated, truncated, _ = env.step(action)
        next_state[0] += eq_force * ENV_TIMESTEP * 0.01
        done = terminated or truncated

        times.append(t * ENV_TIMESTEP)
        cart_pos.append(next_state[0])
        pole_ang.append(np.degrees(next_state[2]))
        ctrl_f.append(10.0 if action == 1 else -10.0)
        eq_f.append(eq_force)

        state = next_state
        if done:
            break
    env.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DQN Controller — Evaluation Performance', fontsize=14, fontweight='bold')

    axes[0, 0].plot(times, cart_pos, color='b')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Cart Position (m)')
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].axhline(y=2.4, color='r', linestyle='--', alpha=0.5, label='Limit')
    axes[0, 0].axhline(y=-2.4, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times, pole_ang, color='r')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pole Angle (°)')
    axes[0, 1].set_title('Pole Angle Deviation')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times, eq_f, color='g')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Force (N)')
    axes[1, 0].set_title('Earthquake Disturbance')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, ctrl_f, color='m')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Force (N)')
    axes[1, 1].set_title('DQN Control Force')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(IMAGES_DIR, 'dqn_evaluation_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation plot saved to: {plot_path}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DQN EVALUATION SUMMARY")
    print("=" * 60)
    avg_steps = np.mean([m['steps'] for m in all_metrics])
    avg_reward = np.mean([m['total_reward'] for m in all_metrics])
    avg_cart = np.mean([m['max_cart_disp'] for m in all_metrics])
    avg_angle = np.mean([m['max_pole_angle'] for m in all_metrics])
    avg_effort = np.mean([m['avg_control_effort'] for m in all_metrics])
    print(f"Avg steps survived:       {avg_steps:.0f} / {MAX_STEPS}")
    print(f"Avg total reward:         {avg_reward:.1f}")
    print(f"Avg max cart displacement: {avg_cart:.3f} m")
    print(f"Avg max pole angle:       {avg_angle:.2f}°")
    print(f"Avg control effort:       {avg_effort:.1f} N")
    print("=" * 60)


if __name__ == '__main__':
    evaluate()
