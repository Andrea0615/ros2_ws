#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
import serial

ser = serial.Serial('/dev/ttyUSB0', 115200)

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
    cos_theta2 = (x*2 + y2 - L12 - L2*2) / (2*L1*L2)
    theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))

    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return np.degrees(theta1), np.degrees(theta2)

def point_callback(msg):
    x_c, y_c, z_c = msg.point.x, msg.point.y, msg.point.z
    x_a, y_a, z_a = transform_camera_to_arm(x_c, y_c, z_c)

    try:
        j1, j2 = inverse_kinematics(x_a, y_a)
        comando = f"MOVE {j1:.2f} {j2:.2f} 0 0\n"
        ser.write(comando.encode())
        rospy.loginfo(f"Enviado al brazo: {comando.strip()}")
    except Exception as e:
        rospy.logwarn(f"No se pudo calcular IK: {e}")

def main():
    rospy.init_node('brazo_ik_node')
    rospy.Subscriber('/detected_point', PointStamped, point_callback)
    rospy.loginfo("Nodo de control del brazo iniciado.")
    rospy.spin()

if _name_ == '_main_':
    main()
