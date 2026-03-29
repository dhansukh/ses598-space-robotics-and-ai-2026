#!/usr/bin/env python3
"""
Terrain Mapping Mission Controller — Optimized for Evaluation
==============================================================
Autonomous mission: search for cylinder via ArUco marker, orbit-map it
at multiple altitudes (saving images for 3D reconstruction), and land
precisely on top using ArUco-guided visual servoing.

State machine:
  INIT -> TAKEOFF -> SEARCH -> APPROACH -> ORBIT_MAP -> DESCEND_LAND -> COMPLETE

Evaluation metrics tracked:
  - Total mission time
  - Energy consumed (time-based estimate)
  - Landing precision (XY distance from cylinder center)
  - Images saved for 3D model accuracy
"""

import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from px4_msgs.msg import (
    VehicleOdometry, OffboardControlMode, VehicleCommand,
    TrajectorySetpoint
)
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster

from cv_bridge import CvBridge
import cv2
import numpy as np


class TerrainMission(Node):
    """Autonomous terrain mapping and cylinder landing mission."""

    def __init__(self):
        super().__init__('terrain_mission')

        # -- QoS --
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # -- Publishers --
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', pub_qos)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', pub_qos)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', pub_qos)
        self.gimbal_pitch_pub = self.create_publisher(
            Float64, '/model/x500_gimbal_0/command/gimbal_pitch', 10)
        self.gimbal_roll_pub = self.create_publisher(
            Float64, '/model/x500_gimbal_0/command/gimbal_roll', 10)
        self.gimbal_yaw_pub = self.create_publisher(
            Float64, '/model/x500_gimbal_0/command/gimbal_yaw', 10)

        # -- Visualization publishers --
        self.path_pub = self.create_publisher(Path, '/drone/path', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.flight_path = Path()
        self.flight_path.header.frame_id = 'map'

        # -- Subscribers --
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_cb, sub_qos)
        self.create_subscription(Image, '/drone_camera', self._image_cb, 10)
        self.create_subscription(CameraInfo, '/drone_camera_info', self._caminfo_cb, 10)

        # -- ArUco detector --
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = 0.8  # meters (matches model.sdf)

        # -- State --
        self.state = 'INIT'
        self.counter = 0
        self.pos = [0.0, 0.0, 0.0]  # NED
        self.bridge = CvBridge()

        # Camera intrinsics
        self.fx = self.fy = self.cx_cam = self.cy_cam = None
        self.img_w = self.img_h = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(5)

        # Timing
        self.mission_start = None

        # -- Search --
        self.search_alt = -20.0          # 20m AGL for better ArUco visibility
        self.search_waypoints = []
        self.search_wp_idx = 0
        self._generate_search_waypoints()

        # -- ArUco detection --
        self.marker_detected = False
        self.marker_pixel = None         # (cx, cy) in image
        self.marker_world = None         # (x, y) NED world
        self.detection_count = 0
        self.detection_threshold = 3     # frames to confirm
        self.candidate_positions = []

        # -- Orbit mapping --
        self.orbit_radius = 6.0
        self.orbit_altitudes = [-20.0, -16.0, -13.0]  # 3 altitudes (NED)
        self.orbit_alt_idx = 0
        self.orbit_speed = 0.04          # rad/step at 10Hz
        self.orbit_theta = 0.0
        self.orbit_revolutions = 0.0
        self.orbit_target_revs = 1.0     # 1 rev per altitude
        self.orbit_center = None
        self.orbit_yaw = 0.0             # current yaw during orbit

        # -- Image saving for 3D reconstruction --
        self.image_save_dir = os.path.expanduser('~/mission_images')
        os.makedirs(self.image_save_dir, exist_ok=True)
        self.image_counter = 0
        self.save_every_n = 3  # save every 3rd image for denser coverage
        self.intrinsics_saved = False

        # -- Descent / landing --
        self.descend_target = -12.0      # start descent altitude
        self.landing_corrections = [0.0, 0.0]  # dx, dy visual servoing
        self.marker_visible_during_descent = False
        self.descent_stabilize_time = None  # timer for hover-stabilize before LAND

        # -- Control loop 10 Hz --
        self.timer = self.create_timer(0.1, self._control_loop)
        self.get_logger().info('TerrainMission node initialized (ArUco-optimized).')

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg):
        self.pos = [msg.position[0], msg.position[1], msg.position[2]]
        self._publish_viz()

    def _publish_viz(self):
        """Publish TF and Path for RViz visualization."""
        now = self.get_clock().now().to_msg()

        # NED → ENU for RViz: swap x↔y, negate z
        enu_x = float(self.pos[1])
        enu_y = float(self.pos[0])
        enu_z = float(-self.pos[2])

        # TF: map → base_link
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = enu_x
        t.transform.translation.y = enu_y
        t.transform.translation.z = enu_z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        # Path: accumulate trail
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = 'map'
        pose.pose.position.x = enu_x
        pose.pose.position.y = enu_y
        pose.pose.position.z = enu_z
        pose.pose.orientation.w = 1.0
        self.flight_path.poses.append(pose)
        self.flight_path.header.stamp = now
        self.path_pub.publish(self.flight_path)

    def _caminfo_cb(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx_cam = msg.k[2]
            self.cy_cam = msg.k[5]
            self.img_w = msg.width
            self.img_h = msg.height
            self.camera_matrix = np.array([
                [self.fx, 0, self.cx_cam],
                [0, self.fy, self.cy_cam],
                [0, 0, 1]], dtype=np.float64)
            self.get_logger().info(
                f'Camera: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx_cam:.1f} cy={self.cy_cam:.1f} {self.img_w}x{self.img_h}')

    def _image_cb(self, msg):
        """Process gimbal camera image for ArUco detection."""
        if self.state not in ('SEARCH', 'APPROACH', 'ORBIT_MAP', 'DESCEND_LAND'):
            return
        if self.camera_matrix is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        self._process_aruco(frame)

    # ------------------------------------------------------------------ #
    #  ArUco detection and pose estimation
    # ------------------------------------------------------------------ #
    def _process_aruco(self, frame):
        """Detect ArUco markers and estimate world position."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return

        # Use first detected marker
        marker_corners = corners[0][0]  # 4x2 array
        center_px = marker_corners.mean(axis=0)
        self.marker_pixel = (int(center_px[0]), int(center_px[1]))

        # Pose estimation using solvePnP
        obj_pts = np.array([
            [-self.marker_size/2,  self.marker_size/2, 0],
            [ self.marker_size/2,  self.marker_size/2, 0],
            [ self.marker_size/2, -self.marker_size/2, 0],
            [-self.marker_size/2, -self.marker_size/2, 0]], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts, marker_corners.astype(np.float32),
            self.camera_matrix, self.dist_coeffs)

        if not success:
            return

        # tvec is marker position in camera frame (meters)
        cam_x, cam_y, cam_z = tvec.flatten()

        # During SEARCH: accumulate position estimates for confirmation
        if self.state == 'SEARCH' and not self.marker_detected:
            world_xy = self._cam_to_world(cam_x, cam_y, cam_z)
            if world_xy is not None:
                self.candidate_positions.append(world_xy)
                self.detection_count += 1
                self.get_logger().info(
                    f'ArUco marker spotted (#{self.detection_count}) '
                    f'at cam_z={cam_z:.1f}m, est world=({world_xy[0]:.1f}, {world_xy[1]:.1f})')

                if self.detection_count >= self.detection_threshold:
                    xs = [p[0] for p in self.candidate_positions[-self.detection_threshold:]]
                    ys = [p[1] for p in self.candidate_positions[-self.detection_threshold:]]
                    self.marker_world = (sum(xs)/len(xs), sum(ys)/len(ys))
                    self.marker_detected = True
                    self.get_logger().info(
                        f'*** CYLINDER CONFIRMED at ({self.marker_world[0]:.2f}, '
                        f'{self.marker_world[1]:.2f}) ***')

        # During DESCEND_LAND: compute visual servoing corrections
        if self.state == 'DESCEND_LAND':
            self.marker_visible_during_descent = True
            # Pixel offset from image center → position correction
            err_x = float(center_px[0] - self.cx_cam) / float(self.fx)
            err_y = float(center_px[1] - self.cy_cam) / float(self.fy)
            
            # The marker is exactly at the cylinder top (-10m NED)
            cylinder_top_ned = -10.0
            h = float(abs(self.pos[2] - cylinder_top_ned))  # altitude above marker
            
            # Strong visual servoing gain for precision landing
            # Note: Camera Y points Backwards (-X), Camera X points Right (+Y)
            self.landing_corrections = [-err_y * h * 0.5, err_x * h * 0.5]

        # During ORBIT_MAP: save images for 3D reconstruction
        if self.state == 'ORBIT_MAP':
            # Save camera intrinsics once
            if not self.intrinsics_saved:
                intrinsics_file = os.path.join(self.image_save_dir, 'camera_intrinsics.txt')
                with open(intrinsics_file, 'w') as f:
                    f.write(f'# Camera Intrinsics\n')
                    f.write(f'fx {float(self.fx):.4f}\n')
                    f.write(f'fy {float(self.fy):.4f}\n')
                    f.write(f'cx {float(self.cx_cam):.4f}\n')
                    f.write(f'cy {float(self.cy_cam):.4f}\n')
                    f.write(f'width {self.img_w}\n')
                    f.write(f'height {self.img_h}\n')
                self.intrinsics_saved = True
                self.get_logger().info(f'Camera intrinsics saved to {intrinsics_file}')

            self.image_counter += 1
            if self.image_counter % self.save_every_n == 0:
                fname = os.path.join(
                    self.image_save_dir,
                    f'orbit_{self.orbit_alt_idx}_{self.image_counter:05d}.jpg')
                cv2.imwrite(fname, frame)
                # Save full 6-DOF camera pose (position + yaw)
                pose_file = fname.replace('.jpg', '_pose.txt')
                with open(pose_file, 'w') as f:
                    f.write(f'# x y z yaw(rad) alt_idx orbit_theta\n')
                    f.write(f'{float(self.pos[0]):.6f} {float(self.pos[1]):.6f} '
                            f'{float(self.pos[2]):.6f} {self.orbit_yaw:.6f} '
                            f'{self.orbit_alt_idx} {self.orbit_theta:.6f}\n')

    def _cam_to_world(self, cam_x, cam_y, cam_z):
        """Convert camera-frame position to NED world coordinates.
        With gimbal pitched -90° (looking straight down):
          cam_z → ground distance, cam_x → drone right (NED Y), cam_y → drone forward (NED X)
        """
        if cam_z < 0.5:
            return None
        # The marker is on top of the cylinder; position in world is:
        world_x = self.pos[0] + cam_y
        world_y = self.pos[1] + cam_x
        return (world_x, world_y)

    # ------------------------------------------------------------------ #
    #  Search pattern (expanding square, efficient for nearby targets)
    # ------------------------------------------------------------------ #
    def _generate_search_waypoints(self):
        """Expanding square centered on origin, step=5m."""
        wps = [(0.0, 0.0)]
        step = 5.0
        x, y = 0.0, 0.0
        for i in range(1, 8):
            d = step * i
            x += d;  wps.append((x, y))
            y += d;  wps.append((x, y))
            x -= 2*d; wps.append((x, y))
            y -= 2*d; wps.append((x, y))
            x += 2*d; wps.append((x, y))
        self.search_waypoints = wps

    # ------------------------------------------------------------------ #
    #  Main control loop (10 Hz)
    # ------------------------------------------------------------------ #
    def _control_loop(self):
        self._publish_offboard_mode()

        if self.counter == 10:
            self._engage_offboard()
            self._arm()
            self.mission_start = time.time()
            self.get_logger().info('Mission started.')

        handlers = {
            'INIT': self._state_init,
            'TAKEOFF': self._state_takeoff,
            'SEARCH': self._state_search,
            'APPROACH': self._state_approach,
            'ORBIT_MAP': self._state_orbit_map,
            'DESCEND_LAND': self._state_descend_land,
            'COMPLETE': self._state_complete,
        }
        handlers.get(self.state, lambda: None)()
        self.counter += 1

    # ------------------------------------------------------------------ #
    #  States
    # ------------------------------------------------------------------ #
    def _state_init(self):
        if self.camera_matrix is not None:
            self.get_logger().info('Camera ready → TAKEOFF')
            self.state = 'TAKEOFF'
        else:
            self._publish_setpoint(0.0, 0.0, 0.0)

    def _state_takeoff(self):
        self._set_gimbal_down()
        self._publish_setpoint(0.0, 0.0, self.search_alt)
        if self._at_pos(0.0, 0.0, self.search_alt, 1.5):
            self.get_logger().info(f'At {-self.search_alt:.0f}m → SEARCH')
            self.state = 'SEARCH'
            self.search_wp_idx = 0

    def _state_search(self):
        self._set_gimbal_down()
        if self.marker_detected:
            self.get_logger().info('ArUco confirmed → APPROACH')
            self.state = 'APPROACH'
            return
        if self.search_wp_idx >= len(self.search_waypoints):
            self.search_wp_idx = 0
            self.get_logger().warn('Search exhausted, restarting.')
        wx, wy = self.search_waypoints[self.search_wp_idx]
        self._publish_setpoint(wx, wy, self.search_alt)
        if self._at_pos(wx, wy, self.search_alt, 2.0):
            self.search_wp_idx += 1

    def _state_approach(self):
        self._set_gimbal_down()
        if self.marker_world is None:
            self.state = 'SEARCH'
            return
        tx, ty = self.marker_world
        self._publish_setpoint(tx, ty, self.search_alt)
        if self._at_pos(tx, ty, self.search_alt, 1.5):
            self.get_logger().info('Above cylinder → ORBIT_MAP')
            self.orbit_center = (tx, ty)
            self.orbit_theta = 0.0
            self.orbit_revolutions = 0.0
            self.orbit_alt_idx = 0
            self.image_counter = 0
            self.state = 'ORBIT_MAP'

    def _state_orbit_map(self):
        """Orbit at multiple altitudes for 3D mapping."""
        cx, cy = self.orbit_center
        alt = self.orbit_altitudes[self.orbit_alt_idx]

        ox = cx + self.orbit_radius * math.cos(self.orbit_theta)
        oy = cy + self.orbit_radius * math.sin(self.orbit_theta)
        yaw = math.atan2(cy - oy, cx - ox)  # face center

        self._publish_setpoint(ox, oy, alt, yaw=yaw)
        self.orbit_yaw = yaw
        self._set_gimbal_pitch(-0.6)  # ~34° down toward cylinder

        self.orbit_theta += self.orbit_speed
        if self.orbit_theta >= 2 * math.pi:
            self.orbit_theta -= 2 * math.pi
            self.orbit_revolutions += 1.0
            self.get_logger().info(
                f'Orbit rev {self.orbit_revolutions:.0f} at alt '
                f'{-alt:.0f}m complete (altitude {self.orbit_alt_idx+1}'
                f'/{len(self.orbit_altitudes)})')

        if self.orbit_revolutions >= self.orbit_target_revs:
            self.orbit_alt_idx += 1
            self.orbit_revolutions = 0.0
            if self.orbit_alt_idx >= len(self.orbit_altitudes):
                self.get_logger().info(
                    f'All orbits complete. {self.image_counter} images processed. → DESCEND_LAND')
                self.state = 'DESCEND_LAND'
                self.descend_target = self.search_alt  # start from current
                self.descend_start_time = time.time()  # Failsafe timer
            else:
                self.get_logger().info(
                    f'Next orbit altitude: {-self.orbit_altitudes[self.orbit_alt_idx]:.0f}m')

    def _state_descend_land(self):
        """Multi-step descent with ArUco visual servoing for precision landing."""
        self._set_gimbal_down()

        # Failsafe: if we spend more than 20 seconds trying to descend, force LAND
        if hasattr(self, 'descend_start_time') and time.time() - self.descend_start_time > 20.0:
            self.get_logger().warn('Descent timeout (20s) reached! Forcing LAND command.')
            self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.state = 'COMPLETE'
            return

        if self.marker_visible_during_descent:
            # Apply visual servoing to current position to track the marker
            adj_x = float(self.pos[0]) + self.landing_corrections[0]
            adj_y = float(self.pos[1]) + self.landing_corrections[1]
            # Update our tracked cylinder center memory
            self.orbit_center = (adj_x, adj_y)
        else:
            # Fallback to the last known cylinder position
            adj_x, adj_y = self.orbit_center

        # Cylinder top ≈ -10.0 in NED
        cylinder_top_ned = -10.0
        hover_above = cylinder_top_ned - 0.5  # 0.5m above top

        current_alt = float(self.pos[2])

        # Step-down descent: reduce altitude by 1m per stabilization
        if current_alt < hover_above + 0.5:  # close to target hover
            # Stabilize above cylinder top
            self._publish_setpoint(adj_x, adj_y, hover_above)

            if self._at_pos(adj_x, adj_y, hover_above, tol=0.5):
                if self.descent_stabilize_time is None:
                    self.descent_stabilize_time = time.time()
                    self.get_logger().info(
                        f'Stabilizing above cylinder top at '
                        f'({float(self.pos[0]):.2f}, {float(self.pos[1]):.2f}, '
                        f'{float(self.pos[2]):.2f})')

                # Wait 3 seconds at hover to ensure stable position
                if time.time() - self.descent_stabilize_time > 3.0:
                    self.get_logger().info('Stable hover achieved → LAND command')
                    self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                    self.state = 'COMPLETE'
        else:
            # Descend gradually: step down 2m at a time
            target_alt = max(current_alt + 2.0, hover_above)
            self._publish_setpoint(adj_x, adj_y, target_alt)

        # Reset marker visibility flag each iteration
        self.marker_visible_during_descent = False

    def _state_complete(self):
        if self.mission_start is not None:
            dt = time.time() - self.mission_start
            self.get_logger().info('=== MISSION COMPLETE ===')
            self.get_logger().info(f'Duration: {dt:.1f} s')
            self.get_logger().info(f'Energy: {dt * 0.5:.1f} units')

            if self.orbit_center is not None:
                cx, cy = self.orbit_center
                err = math.sqrt((self.pos[0]-cx)**2 + (self.pos[1]-cy)**2)
                self.get_logger().info(
                    f'Position: ({float(self.pos[0]):.2f}, {float(self.pos[1]):.2f}, {float(self.pos[2]):.2f})')
                self.get_logger().info(f'Landing error: {err:.3f} m')

            img_count = len([f for f in os.listdir(self.image_save_dir) if f.endswith('.jpg')])
            self.get_logger().info(f'3D mapping images saved: {img_count}')
            self.get_logger().info('========================')
            self.mission_start = None
            # Schedule shutdown after 3 seconds to allow LAND to execute
            self._shutdown_time = time.time()

        # Auto-shutdown after metrics are logged
        if hasattr(self, '_shutdown_time') and time.time() - self._shutdown_time > 3.0:
            self.get_logger().info('Mission node shutting down.')
            self.timer.cancel()
            raise SystemExit(0)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _at_pos(self, x, y, z, tol=1.0):
        return math.sqrt((self.pos[0]-x)**2 + (self.pos[1]-y)**2 + (self.pos[2]-z)**2) < tol

    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = msg.acceleration = msg.attitude = msg.body_rate = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_pub.publish(msg)

    def _publish_setpoint(self, x, y, z, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.traj_pub.publish(msg)

    def _send_vehicle_command(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.param1, msg.param2 = float(p1), float(p2)
        msg.command = command
        msg.target_system = msg.target_component = 1
        msg.source_system = msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.cmd_pub.publish(msg)

    def _arm(self):
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)
        self.get_logger().info('Armed')

    def _engage_offboard(self):
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
        self.get_logger().info('Offboard mode')

    def _set_gimbal_down(self):
        self._set_gimbal_pitch(-1.5708)

    def _set_gimbal_pitch(self, pitch):
        p = Float64(); p.data = float(pitch); self.gimbal_pitch_pub.publish(p)
        r = Float64(); r.data = 0.0; self.gimbal_roll_pub.publish(r)


def main(args=None):
    rclpy.init(args=args)
    node = TerrainMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Mission interrupted.')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
