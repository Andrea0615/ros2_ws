import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from odometry.msg import WheelInfo
from controller import angulo_ackermann, find_look_ahead_point, generar_ruta_prioritaria, find_stopping_point, robot_stop
from efk import compute_F, predict_state
from utils import compute_quaternion, RPMReader, IMUListener, CoordinatesListener, SynchronizedData, initialize_serial, send_rpm_command
import time

# Parámetros de conexión serial (ajusta según tu sistema: 'COM6' para Windows o '/dev/ttyUSB0' para Linux/Mac)
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
TIMEOUT = 1

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.wheel_pub = self.create_publisher(WheelInfo, '/wheel_setter', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

        self.start_time = time.time()  # Guardar el tiempo de inicio
        
        # Agregar clases para lectura de sensores
        self.sync_data = SynchronizedData()
        self.imu_listener = IMUListener(self.sync_data)
        self.rpm_listener = RPMReader(self.sync_data, port=SERIAL_PORT)
        self.dt = 0.05
        self.lookAheadDist = 1.5
        self.desiredSpeed = 0.4
        self.L = 0.89
        self.wheelDiameter = 0.24
        self.wheelCircumference = np.pi * self.wheelDiameter

        # Estado inicial
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = np.radians(0.0)
        self.xhat = np.array([self.odom_x, self.odom_y, self.odom_theta, 0.0, 0.0, 0.0])
        self.P = np.identity(6) * 0.01
        self.Q = np.diag([0.001, 0.001, 0.0005, 0.001, 0.001, 0.0001])
        self.R = np.diag([0.02, 0.02, 0.01, self.rpm_listener["linear_velocity_std"] + self.imu_listener["std_dev"]["accel_x"], self.imu_listener["std_dev"]["gyro_z"]])

        self.idxWaypoint = 0
        self.waypoints = []  # Dynamic waypoint list
        self.coordenadas_camara = CoordinatesListener()

        # Inicializar la conexión serial para enviar comandos de RPM
        self.ser = initialize_serial(SERIAL_PORT, BAUD_RATE, TIMEOUT)

    def timer_callback(self):
        elapsed_time = time.time() - self.start_time  # Calculamos el tiempo transcurrido para detenerse antes de los 10min

        # Get next waypoint
        piedras = self.coordenadas_camara.get_new_coords()
        piedra_x = piedras[0]
        piedra_dist = piedras[1]
        stopping_point = find_stopping_point(piedra_x, piedra_dist, self.xhat[0], self.xhat[1], self.xhat[2])
        if robot_stop(stopping_point, self.odom_x, self.odom_y):
            send_rpm_command(self.ser, 0, 0, 0, 0)  # Detiene las 4 ruedas
            self.get_logger().info("Robot detenido en el punto de parada.")
            return  # Opcional: aquí puede ir la lógica de recolección

        next_point = generar_ruta_prioritaria(stopping_point, use_push_front=False)

        if next_point is None and not self.waypoints:
            self.get_logger().info("No more waypoints or stones.")
            return

        if next_point:
            self.waypoints.append(next_point)
            # Keep waypoints list manageable (e.g., remove old points)
            if len(self.waypoints) > 100:
                self.waypoints = self.waypoints[-50:]

        if not self.waypoints:
            return

        # Convert to NumPy array for find_look_ahead_point
        waypoints_array = np.array(self.waypoints)

        # Compute look-ahead point
        lookX, lookY, self.idxWaypoint = find_look_ahead_point(
            self.odom_x, self.odom_y, waypoints_array, self.idxWaypoint, self.lookAheadDist
        )

        # Remove waypoints behind the robot
        if self.idxWaypoint > 0:
            self.waypoints = self.waypoints[self.idxWaypoint:]
            self.idxWaypoint = 0

        # Control calculations
        dx = lookX - self.odom_x
        dy = lookY - self.odom_y
        L_d = np.hypot(dx, dy)
        alpha = np.arctan2(dy, dx) - self.odom_theta
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        kappa = 2 * np.sin(alpha) / max(L_d, 1e-9)
        delta = np.arctan(kappa * self.L)
        b, v_interior = angulo_ackermann(delta, self.desiredSpeed)

        # Cálculo de RPM basados en la circunferencia de la llanta
        rpm_exterior = (self.desiredSpeed / self.wheelCircumference) * 60
        rpm_interior = (v_interior / self.wheelCircumference) * 60

        # Get synchronized sensor data
        sensor_data = self.sync_data.get_latest_data()
        real_RPM = sensor_data['rpm']
        imu_data = sensor_data['imu']
        real_velocity = self.rpm_listener["linear_velocity"]
        RPM = (self.desiredSpeed / self.wheelCircumference) * 60

        # Si lo que se desea es enviar el comando de RPM calculado, se puede elegir entre:
        #  - En línea recta, las 4 llantas con rpm_exterior
        #  - Al girar, se asignan las RPM de acuerdo al lado interior/exterior
        if delta > 0:
            # Giro a la izquierda: llantas izquierdas interiores
            rpm_left = rpm_interior
            rpm_right = rpm_exterior
        elif delta < 0:
            # Giro a la derecha: llantas derechas interiores
            rpm_left = rpm_exterior
            rpm_right = rpm_interior
        else:
            rpm_left = rpm_right = rpm_exterior

        # Enviar comando de RPM a las llantas (se asume:
        #   M1 y M3: llantas izquierdas,
        #   M2 y M4: llantas derechas)
        send_rpm_command(self.ser, rpm_left, rpm_right, rpm_left, rpm_right)

        # Simulated measurement with noise
        noise = np.random.normal(0, np.sqrt(np.diag(self.R)))
        z = np.array([
            self.odom_x,
            self.odom_y,
            self.odom_theta,
            real_velocity + imu_data['accel_filtered']['x'],
            imu_data['gyro_filtered']['z']
        ]) + noise

        # Update odometry
        omega = (real_velocity / self.L) * np.tan(delta)
        self.odom_x += real_velocity * np.cos(self.odom_theta) * self.dt
        self.odom_y += real_velocity * np.sin(self.odom_theta) * self.dt
        self.odom_theta += omega * self.dt

        # EKF update
        u = np.array([real_velocity, delta])
        xhat_pred = predict_state(self.xhat, u, imu_data, self.L, self.dt)
        F = compute_F(self.xhat, u, imu_data, self.dt)
        P_pred = F @ self.P @ F.T + self.Q
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.xhat = xhat_pred + K @ (z - H @ xhat_pred)
        self.P = (np.eye(6) - K @ H) @ P_pred

        # Preparar y publicar el mensaje de odometría
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.pose.pose.position.x = self.xhat[0]
        odom_msg.pose.pose.position.y = self.xhat[1]
        odom_msg.pose.pose.position.z = 0

        quaternion = compute_quaternion(self.xhat[2])
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]

        # Preparar y publicar el mensaje de los datos de las ruedas
        wheel_msg = WheelInfo()
        wheel_msg.v_interior = v_interior
        wheel_msg.desired_speed = self.desiredSpeed
        wheel_msg.delta = delta
        wheel_msg.beta = b

        self.odom_pub.publish(odom_msg)
        self.wheel_pub.publish(wheel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
