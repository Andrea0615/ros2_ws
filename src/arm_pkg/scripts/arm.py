#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PointStamped
import serial

def transform_camera_to_arm(x_c, y_c, z_c):
    T = np.array([
        [1, 0, 0, -0.10],
        [0, 1, 0,  0.00],
        [0, 0, 1, -0.15],
        [0, 0, 0,  1.0]
    ])
    p_cam = np.array([x_c, y_c, z_c, 1])
    p_arm = T @ p_cam
    return p_arm[:3]

def inverse_kinematics(x, y, L1=0.1, L2=0.1):
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))

    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return np.degrees(theta1), np.degrees(theta2)

class ArmController(Node):
    def __init__(self):
        super().__init__('brazo_ik_node')

        # Initialize serial
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.get_logger().info("Serial connection established.")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            self.ser = None

        # Create subscriber
        self.subscription = self.create_subscription(
            PointStamped,
            '/detected_point',
            self.point_callback,
            10  # QoS depth
        )
        self.subscription  # prevent unused variable warning

        self.get_logger().info("Arm control node started.")

    def point_callback(self, msg):
        if self.ser is None:
            self.get_logger().warn("Serial not available. Skipping message.")
            return

        x_c, y_c, z_c = msg.point.x, msg.point.y, msg.point.z
        x_a, y_a, z_a = transform_camera_to_arm(x_c, y_c, z_c)

        try:
            j1, j2 = inverse_kinematics(x_a, y_a)
            comando = f"MOVE {j1:.2f} {j2:.2f} 0 0\n"
            try:
                self.ser.write(comando.encode())
                self.get_logger().info(f"Sent to arm: {comando.strip()}")
            except serial.SerialException as e:
                self.get_logger().error(f"Serial communication error: {e}")
        except Exception as e:
            self.get_logger().warn(f"Could not compute IK: {e}")

def main(args=None):
    rclpy.init(args=args)

    arm_controller = ArmController()

    rclpy.spin(arm_controller)

    # Clean up
    if arm_controller.ser is not None:
        arm_controller.ser.close()

    arm_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
