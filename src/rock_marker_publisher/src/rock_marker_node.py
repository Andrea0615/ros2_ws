#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
import yaml
import os

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray

class RockMarkerPublisher(Node):
    def __init__(self):
        super().__init__("rock_marker_publisher")

        # Publisher for markers
        self.marker_pub = self.create_publisher(Marker, "/rock_markers", 10)

        # Subscriber for rock coordinates
        self.create_subscription(Float32MultiArray, "/detected_rocks", self.rock_callback, 10)

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Storage
        self.rock_id = 0
        self.rocks = []
        self.storage_path = self.declare_parameter("storage_path", "rocks.yaml").get_parameter_value().string_value

        # Shutdown callback
        self.add_on_shutdown(self.save_rocks_to_file)

        self.load_and_publish_existing()

    def rock_callback(self, msg):
        x_in, y_in = msg.data[0], msg.data[1]

        point_in = PointStamped()
        point_in.header.frame_id = "camera_link"
        point_in.header.stamp = self.get_clock().now().to_msg()
        point_in.point.x = x_in
        point_in.point.y = y_in
        point_in.point.z = 0.0

        try:
            transform = self.tf_buffer.lookup_transform("map", point_in.header.frame_id, rclpy.time.Time())
            point_out = tf2_geometry_msgs.do_transform_point(point_in, transform)
        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return

        rock_coords = {"x": point_out.point.x, "y": point_out.point.y}
        self.rocks.append(rock_coords)
        self.publish_marker(rock_coords["x"], rock_coords["y"])

    def publish_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rocks"
        marker.id = self.rock_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Add lifetime here ➔ marker disappears after 30 seconds (or change the value)
        from builtin_interfaces.msg import Duration
        marker.lifetime = Duration(sec=30)

        self.marker_pub.publish(marker)
        self.rock_id += 1

    def save_rocks_to_file(self):
        with open(self.storage_path, 'w') as f:
            yaml.dump({'rocks': self.rocks}, f)
        self.get_logger().info(f"Saved {len(self.rocks)} rocks to {self.storage_path}")

    def load_and_publish_existing(self):
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'rocks' in data:
                for rock in data['rocks']:
                    self.publish_marker(rock['x'], rock['y'])
                self.rock_id = len(data['rocks'])
                self.get_logger().info(f"Re-published {self.rock_id} saved rocks")

def main(args=None):
    rclpy.init(args=args)  # Start ROS2 communication
    node = RockMarkerPublisher()  # Create the node
    rclpy.spin(node)  # Keep it alive and listen for messages
    node.destroy_node()  # Clean up the node
    rclpy.shutdown()  # Shut down ROS2

if __name__ == "__main__":
    main()
