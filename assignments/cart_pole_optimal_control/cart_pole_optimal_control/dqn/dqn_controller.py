#!/usr/bin/env python3
"""
DQN Controller — ROS2 Node for Cart-Pole Control.

Loads a pre-trained DQN model and uses it to control the cart-pole system
in the Gazebo simulation. Mirrors the LQR controller interface: subscribes
to joint states and earthquake forces, publishes cart force commands.
"""

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

import torch
import sys

# Allow importing dqn_agent from the same package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dqn_agent import QNetwork


class CartPoleDQNController(Node):
    def __init__(self):
        super().__init__('cart_pole_dqn_controller')

        # System parameters (same as LQR controller)
        self.M = 1.0   # Cart mass (kg)
        self.m = 1.0   # Pole mass (kg)
        self.L = 1.0   # Pole length (m)
        self.g = 9.81  # Gravity (m/s²)

        # DQN configuration
        self.state_dim = 5   # [x, x_dot, theta, theta_dot, earthquake_force]
        self.action_dim = 2  # Discrete: left (0) or right (1)
        self.force_magnitude = 50.0  # Force magnitude for each action

        # Load trained model
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, 'dqn_cartpole_earthquake.pth')

        self.device = torch.device('cpu')
        self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)

        try:
            self.q_network.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.q_network.eval()
            self.get_logger().info(f'DQN model loaded from {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load DQN model: {e}')
            self.get_logger().warn('Controller will use random actions!')

        # State
        self.x = np.zeros(4)
        self.state_initialized = False
        self.earthquake_force = 0.0
        self.last_control = 0.0
        self.control_count = 0

        # Data storage for plotting
        self.time_steps = deque()
        self.cart_positions = deque()
        self.pole_angles = deque()
        self.control_forces = deque()
        self.earthquake_forces = deque()
        self.start_time = None

        # Publishers & Subscribers (same topics as LQR)
        self.cart_cmd_pub = self.create_publisher(
            Float64, '/model/cart_pole/joint/cart_to_base/cmd_force', 10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/world/empty/model/cart_pole/joint_state',
            self.joint_state_callback,
            10
        )

        self.earthquake_sub = self.create_subscription(
            Float64, '/earthquake_force', self.earthquake_callback, 10
        )

        # Control loop timer (100 Hz, same as LQR)
        self.timer = self.create_timer(0.01, self.control_loop)

        self.MAX_SIMULATION_TIME = 120.0

        self.get_logger().info('Cart-Pole DQN Controller initialized')

    def joint_state_callback(self, msg):
        """Update state estimate from joint states."""
        try:
            cart_idx = msg.name.index('cart_to_base')
            pole_idx = msg.name.index('pole_joint')

            self.x = np.array([
                msg.position[cart_idx],
                msg.velocity[cart_idx],
                msg.position[pole_idx],
                msg.velocity[pole_idx]
            ])

            if not self.state_initialized:
                self.get_logger().info(
                    f'Initial state: cart_pos={self.x[0]:.3f}, '
                    f'cart_vel={self.x[1]:.3f}, '
                    f'pole_angle={self.x[2]:.3f}, '
                    f'pole_vel={self.x[3]:.3f}'
                )
                self.state_initialized = True
                self.start_time = self.get_clock().now().nanoseconds / 1e9

        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'Failed to process joint states: {e}')

    def earthquake_callback(self, msg):
        """Store latest earthquake force value."""
        self.earthquake_force = msg.data

    def select_action(self, state_augmented):
        """Use the DQN to select an action."""
        state_tensor = torch.FloatTensor(state_augmented).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    def print_metrics(self):
        """Print performance metrics after simulation ends."""
        duration = self.time_steps[-1] if self.time_steps else 0.0
        max_cart_disp = max(map(abs, self.cart_positions), default=0.0)
        max_pole_dev = max(map(abs, self.pole_angles), default=0.0)
        avg_control = np.mean(np.abs(list(self.control_forces))) if self.control_forces else 0.0

        self.get_logger().info(f"Duration of stable operation: {duration:.2f} s")
        self.get_logger().info(f"Maximum cart displacement: {max_cart_disp:.3f} m")
        self.get_logger().info(f"Maximum pendulum angle deviation: {max_pole_dev:.3f}°")
        self.get_logger().info(f"Average control effort: {avg_control:.3f} N")

    def control_loop(self):
        """Compute and apply DQN control."""
        try:
            if not self.state_initialized:
                return

            # Build augmented state
            state_augmented = np.append(self.x, self.earthquake_force)

            # DQN action selection
            action = self.select_action(state_augmented)

            # Map discrete action to continuous force
            # action 0 → push left, action 1 → push right
            force = self.force_magnitude if action == 1 else -self.force_magnitude

            # Publish force command
            msg = Float64()
            msg.data = force
            self.cart_cmd_pub.publish(msg)

            self.last_control = force
            self.control_count += 1

            # Log data
            current_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
            self.time_steps.append(current_time)
            self.cart_positions.append(self.x[0])
            self.pole_angles.append(np.degrees(self.x[2]))
            self.control_forces.append(force)
            self.earthquake_forces.append(self.earthquake_force)

            # Termination conditions (same as LQR)
            if (
                abs(self.x[0]) > 2.5
                or abs(self.x[2]) > np.radians(45)
                or current_time >= self.MAX_SIMULATION_TIME
            ):
                self.get_logger().warn(
                    f"Simulation ended: cart_x={self.x[0]:.2f}m, "
                    f"pole_angle={np.degrees(self.x[2]):.2f}°, "
                    f"duration={current_time:.2f}s"
                )
                self.print_metrics()
                self.plot_results()
                rclpy.shutdown()
                return

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

    def plot_results(self):
        """Generate plots for analysis."""
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('DQN Controller — Simulation Results', fontsize=14, fontweight='bold')

        plt.subplot(2, 2, 1)
        plt.plot(list(self.time_steps), list(self.cart_positions), color='b')
        plt.xlabel('Time (s)')
        plt.ylabel('Cart Position (m)')
        plt.title('Cart Position')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(list(self.time_steps), list(self.pole_angles), color='r')
        plt.xlabel('Time (s)')
        plt.ylabel('Pole Angle (°)')
        plt.title('Pole Angle')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(list(self.time_steps), list(self.earthquake_forces), color='g')
        plt.xlabel('Time (s)')
        plt.ylabel('Earthquake Force (N)')
        plt.title('Earthquake Disturbance')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(list(self.time_steps), list(self.control_forces), color='m')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Force (N)')
        plt.title('DQN Control Force')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        images_dir = os.path.join(project_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        plot_path = os.path.join(images_dir, 'dqn_simulation_results.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f'Results plot saved to: {plot_path}')


def main(args=None):
    rclpy.init(args=args)
    controller = CartPoleDQNController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
