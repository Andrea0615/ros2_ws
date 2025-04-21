#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
import yaml
import os
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray  # Assuming rocks come as [x, y] in camera/base frame

class RockMarkerPublisher:
    def __init__(self):
        rospy.init_node("rock_marker_publisher")

        # Publisher for markers
        self.marker_pub = rospy.Publisher("/rock_markers", Marker, queue_size=10)

        # Subscriber for rock coordinates
        rospy.Subscriber("/detected_rocks", Float32MultiArray, self.rock_callback)

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Storage
        self.rock_id = 0
        self.rocks = []
        self.storage_path = rospy.get_param("~storage_path", "rocks.yaml")

        rospy.on_shutdown(self.save_rocks_to_file)
        rospy.sleep(1)
        self.load_and_publish_existing()

    def rock_callback(self, msg):
        x_in, y_in = msg.data[0], msg.data[1]

        point_in = PointStamped()
        point_in.header.frame_id = "camera_link"  # Adjust if needed
        point_in.header.stamp = rospy.Time.now()
        point_in.point.x = x_in
        point_in.point.y = y_in
        point_in.point.z = 0.0

        try:
            transform = self.tf_buffer.lookup_transform("map", point_in.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            point_out = tf2_geometry_msgs.do_transform_point(point_in, transform)
        except Exception as e:
            rospy.logwarn("Transform failed: %s", e)
            return

        # Save and publish
        rock_coords = {"x": point_out.point.x, "y": point_out.point.y}
        self.rocks.append(rock_coords)
        self.publish_marker(rock_coords["x"], rock_coords["y"])

    def publish_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
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

        self.marker_pub.publish(marker)
        self.rock_id += 1

    def save_rocks_to_file(self):
        with open(self.storage_path, 'w') as f:
            yaml.dump({'rocks': self.rocks}, f)
        rospy.loginfo("Saved %d rocks to %s", len(self.rocks), self.storage_path)

    def load_and_publish_existing(self):
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'rocks' in data:
                for rock in data['rocks']:
                    self.publish_marker(rock['x'], rock['y'])
                self.rock_id = len(data['rocks'])
                rospy.loginfo("Re-published %d saved rocks", self.rock_id)

if __name__ == "__main__":
    try:
        RockMarkerPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
