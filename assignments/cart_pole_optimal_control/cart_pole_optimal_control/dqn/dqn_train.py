#!/usr/bin/env python3
"""
DQN Training Script for Cart-Pole with Earthquake Disturbances.

Trains a DQN agent on OpenAI Gymnasium's CartPole-v1 environment with
simulated earthquake forces injected into the state space. Saves training
progress plots and the trained model.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import gymnasium as gym

# Allow running from the dqn directory directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dqn_agent import DQNAgent

# ─── Configuration ───────────────────────────────────────────────────────────
NUM_EPISODES = 15000
MAX_STEPS_PER_EPISODE = 1000
TARGET_UPDATE_FREQ = 10       # Update target network every N episodes

# Earthquake parameters (matching the ROS2 earthquake_force_generator)
NUM_WAVES = 5
FREQ_RANGE = [0.5, 4.0]
BASE_AMPLITUDE = 15.0
ENV_TIMESTEP = 0.02

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'dqn_cartpole_earthquake.pth')


def make_earthquake_generator():
    """Create a stateful earthquake force generator."""
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


def compute_reward(next_state, earthquake_force):
    """Custom reward function that penalizes pole tilt and cart displacement."""
    cart_position = abs(next_state[0])
    pole_angle = abs(next_state[2])

    base_reward = 1.0
    pole_stability = 1.0 - (2.5 * pole_angle)
    cart_stability = 1.0 - (0.5 * cart_position)

    # Bonus for surviving under strong disturbance
    disturbance_bonus = 0.1 * (abs(earthquake_force) / BASE_AMPLITUDE)

    reward = base_reward + pole_stability + cart_stability + disturbance_bonus
    return max(reward, 0.0)


def moving_average(data, window=100):
    """Compute moving average for smooth trend visualization."""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def train():
    """Main training loop."""
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0] + 1  # 4 states + 1 earthquake force
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    earthquake = make_earthquake_generator()

    # Tracking
    total_rewards = []
    steps_per_episode = []
    epsilon_values = []
    losses = []

    global_step = 0

    print(f"Starting DQN Training: {NUM_EPISODES} episodes")
    print(f"Earthquake: amplitude={BASE_AMPLITUDE}N, freq={FREQ_RANGE}Hz")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    print(f"Plots will be saved to: {IMAGES_DIR}")
    print("-" * 60)

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0
        episode_loss = 0.0
        steps = 0

        for t in range(MAX_STEPS_PER_EPISODE):
            eq_force = earthquake(global_step * ENV_TIMESTEP)

            # Inject earthquake into Gym's force magnitude
            env.unwrapped.force_mag = 10.0 + abs(eq_force) * 0.5

            # Augment state with earthquake force
            state_augmented = np.append(state, eq_force)

            action = agent.select_action(state_augmented, evaluate=False)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = compute_reward(next_state, eq_force)

            next_state_augmented = np.append(next_state, eq_force)
            agent.store_transition(state_augmented, action, reward, next_state_augmented, done)

            loss = agent.train()
            episode_loss += loss if loss else 0.0

            state = next_state
            global_step += 1
            total_reward += reward
            steps += 1

            if done:
                break

        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        total_rewards.append(total_reward)
        steps_per_episode.append(steps)
        epsilon_values.append(agent.epsilon)
        losses.append(episode_loss / max(steps, 1))

        if episode % 500 == 0:
            avg_reward = np.mean(total_rewards[-500:])
            avg_steps = np.mean(steps_per_episode[-500:])
            print(
                f"Episode {episode:5d} | "
                f"Avg Reward: {avg_reward:7.1f} | "
                f"Avg Steps: {avg_steps:5.0f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    # Save model
    agent.save_model(MODEL_SAVE_PATH)
    env.close()

    # ─── Generate Training Progress Plots ─────────────────────────────────
    print("\nGenerating training progress plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN Training Progress — Cart-Pole with Earthquake Disturbance',
                 fontsize=14, fontweight='bold')

    # 1) Total Reward
    ax = axes[0, 0]
    ax.plot(total_rewards, alpha=0.3, color='steelblue', label='Raw')
    ma = moving_average(total_rewards)
    ax.plot(range(99, 99 + len(ma)), ma, color='navy', linewidth=2, label='Moving Avg (100)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Steps per Episode
    ax = axes[0, 1]
    ax.plot(steps_per_episode, alpha=0.3, color='coral', label='Raw')
    ma = moving_average(steps_per_episode)
    ax.plot(range(99, 99 + len(ma)), ma, color='darkred', linewidth=2, label='Moving Avg (100)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode (Survival Time)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Epsilon Decay
    ax = axes[1, 0]
    ax.plot(epsilon_values, color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (ε)')
    ax.set_title('Exploration Rate Decay')
    ax.grid(True, alpha=0.3)

    # 4) Training Loss
    ax = axes[1, 1]
    ax.plot(losses, alpha=0.3, color='orange', label='Raw')
    ma = moving_average(losses)
    ax.plot(range(99, 99 + len(ma)), ma, color='darkorange', linewidth=2, label='Moving Avg (100)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(IMAGES_DIR, 'dqn_training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to: {plot_path}")

    # ─── Print Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total episodes:           {NUM_EPISODES}")
    print(f"Final epsilon:            {agent.epsilon:.4f}")
    print(f"Avg reward (last 1000):   {np.mean(total_rewards[-1000:]):.1f}")
    print(f"Avg steps (last 1000):    {np.mean(steps_per_episode[-1000:]):.0f}")
    print(f"Max reward achieved:      {max(total_rewards):.1f}")
    print(f"Max steps achieved:       {max(steps_per_episode)}")
    print("=" * 60)


if __name__ == '__main__':
    train()
