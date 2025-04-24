#!/usr/bin/env python
import rospy
import math
import serial
import json
from vision_msgs.msg import Detection2DArray

class RockPickerStaticNode:
    def __init__(self):
        rospy.init_node("rock_picker_static_node")

        self.camera_tilt = rospy.get_param("~camera_tilt", -25.3)
        self.camera_height = rospy.get_param("~camera_height", 0.576)
        self.arm_height = rospy.get_param("~arm_height", 0.37)
        self.arm_offset_z = rospy.get_param("~arm_offset_z", 0.14)
        self.fx = rospy.get_param("~focal_length_x", 500.0)
        self.fy = rospy.get_param("~focal_length_y", 500.0)
        self.cx_0 = rospy.get_param("~principal_point_x", 320.0)
        self.cy_0 = rospy.get_param("~principal_point_y", 240.0)
        self.serial_port = rospy.get_param("~serial_port", "/dev/ttyUSB0")

        self.tilt_rad = math.radians(self.camera_tilt)
        self.cos_tilt = math.cos(self.tilt_rad)
        self.sin_tilt = math.sin(self.tilt_rad)

        try:
            self.ser = serial.Serial(self.serial_port, baudrate=115200, timeout=1)
            rospy.loginfo("Connected to RoArm-M2 on {}".format(self.serial_port))
        except serial.SerialException as e:
            rospy.logerr("Failed to connect to serial port {}: {}".format(self.serial_port, e))
            raise

        self.det_sub = rospy.Subscriber("/detections", Detection2DArray, self.detections_callback)

        self.is_picking = False
        self.target_rock = None
        rospy.loginfo("RockPickerStaticNode initialized")

    def detections_callback(self, msg):
        if self.is_picking:
            return

        min_z_ground = float('inf')
        self.target_rock = None
        best_coords = None

        for det in msg.detections:
            cx = det.bbox.center.x
            cy = det.bbox.center.y
            z = det.bbox.center.theta
            if math.isnan(z):
                continue

            x_camera = z * (cx - self.cx_0) / self.fx
            y_camera = z * (cy - self.cy_0) / self.fy
            z_camera = z

            x_ground = x_camera
            y_ground = y_camera * self.cos_tilt - z_camera * self.sin_tilt
            z_ground = y_camera * self.sin_tilt + z_camera * self.cos_tilt

            if z_ground < min_z_ground:
                min_z_ground = z_ground
                self.target_rock = det
                best_coords = (x_ground, y_ground, z_ground)

        if self.target_rock and best_coords:
            self.process_rock(*best_coords)
        else:
            rospy.loginfo("No rocks detected")

    def process_rock(self, x_ground, y_ground, z_ground):
        if self.is_picking:
            return
        self.is_picking = True
        rospy.loginfo("Processing rock")

        arm_x = (y_ground + (self.camera_height - self.arm_height)) * 1000
        arm_y = x_ground * 1000
        arm_z = (z_ground - self.arm_offset_z) * 1000

        rospy.loginfo(f"Rock coordinates (arm frame): x={arm_x:.2f}mm, y={arm_y:.2f}mm, z={arm_z:.2f}mm")

        # Guardar coordenada en JSON
        json_data = [{
            "T": 1041,
            "x": int(round(arm_x)),
            "y": int(round(arm_y)),
            "z": int(round(arm_z)),
            "t": 3.68 #Cerrado
        }]
        try:
            with open("coordenada_brazo.json", "w") as f:
                json.dump(json_data, f, indent=4)
            rospy.loginfo("Saved JSON command to coordenada_brazo.json")
        except Exception as e:
            rospy.logerr(f"Error writing JSON file: {e}")

        # Enviar comando al brazo
        cmd = {
            "T": 1041,
            "x": arm_x,
            "y": arm_y,
            "z": arm_z,
            "t": 1.08,
        }
        self.send_json_cmd(cmd)
        rospy.sleep(2.0)

        cmd["t"] = 3.14
        self.send_json_cmd(cmd)
        rospy.sleep(1.0)

        cmd["t"] = 1.08
        self.send_json_cmd(cmd)
        rospy.sleep(1.0)

        rospy.loginfo("Rock picked")
        self.is_picking = False
        self.target_rock = None
        rospy.sleep(1.0)

    def send_json_cmd(self, cmd):
        try:
            cmd_str = json.dumps(cmd) + "\n"
            self.ser.write(cmd_str.encode('utf-8'))
            rospy.loginfo("Sent JSON: {}".format(cmd_str.strip()))
        except serial.SerialException as e:
            rospy.logerr("Failed to send JSON command: {}".format(e))

    def shutdown(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        rospy.loginfo("RockPickerStaticNode shutdown")

if __name__ == "__main__":
    try:
        node = RockPickerStaticNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
