# utils.py
import numpy as np
import rospy
import tf.transformations as tft
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from threading import Lock, Thread
import time 
import serial 

class ListQueueSimple:
    """
    A thread-safe queue for storing items (e.g., 2D coordinates) with FIFO behavior.
    Supports enqueue, dequeue, isempty, and push_front for prioritization.
    """
    def __init__(self):
        self.items = []
        self.lock = Lock()

    def enqueue(self, item):
        """Add an item to the tail of the queue."""
        with self.lock:
            self.items.append(item)

    def push_front(self, item):
        """Add an item to the head of the queue (for prioritization)."""
        with self.lock:
            self.items.insert(0, item)

    def dequeue(self):
        """Remove and return an item from the head of the queue."""
        with self.lock:
            if self.items:
                return self.items.pop(0)
            return None

    def isempty(self):
        """Check if the queue is empty."""
        with self.lock:
            return len(self.items) == 0

    def size(self):
        """Return the number of items in the queue."""
        with self.lock:
            return len(self.items)

class IMUListener:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=115200, timeout=1): #Cambiar puerto serial
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = self.initialize_serial()

        self.imu_data = {
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            'accel': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'gyro': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'std_dev': {
                'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
                'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0
            }
        }

    def initialize_serial(self):
        try:
            ser = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)
            print(f"Conectado al puerto {self.serial_port} a {self.baud_rate} baudios.")
            time.sleep(2)
            return ser
        except serial.SerialException as e:
            print(f"Error al conectar al puerto {self.serial_port}: {e}")
            return None

    def read_data(self):
        if self.ser is None or not self.ser.is_open:
            print("Error: No hay conexión serial activa.")
            return

        try:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                data = line.split(',')
                label = data[0]

                if label == "YPR":
                    self.imu_data['yaw'] = float(data[1])
                    self.imu_data['pitch'] = float(data[2])
                    self.imu_data['roll'] = float(data[3])

                elif label == "ACC":
                    self.imu_data['accel']['x'] = float(data[1])
                    self.imu_data['accel']['y'] = float(data[2])
                    self.imu_data['accel']['z'] = float(data[3])

                elif label == "GYRO":
                    self.imu_data['gyro']['x'] = float(data[1])
                    self.imu_data['gyro']['y'] = float(data[2])
                    self.imu_data['gyro']['z'] = float(data[3])

                elif label == "STD_ACC":
                    self.imu_data['std_dev']['accel_x'] = float(data[1])
                    self.imu_data['std_dev']['accel_y'] = float(data[2])
                    self.imu_data['std_dev']['accel_z'] = float(data[3])

                elif label == "STD_GYRO":
                    self.imu_data['std_dev']['gyro_x'] = float(data[1])
                    self.imu_data['std_dev']['gyro_y'] = float(data[2])
                    self.imu_data['std_dev']['gyro_z'] = float(data[3])

                elif label == "STD_YPR":
                    self.imu_data['std_dev']['yaw'] = float(data[1])
                    self.imu_data['std_dev']['pitch'] = float(data[2])
                    self.imu_data['std_dev']['roll'] = float(data[3])
        except (UnicodeDecodeError, ValueError, IndexError) as e:
            print(f"Error procesando línea: {line} -> {e}")

    def get_data(self):
        return self.imu_data


def compute_quaternion(theta):
    return tft.quaternion_from_euler(0, 0, theta)


class RPMReader:
    def __init__(self, window_size=50, alpha=0.1):
        self.rpm_fl = 0.0
        self.rpm_fr = 0.0
        self.rpm_rl = 0.0
        self.rpm_rr = 0.0

        self.window_size = window_size
        self.alpha = alpha

        self.buffer_fl = []
        self.buffer_fr = []
        self.buffer_rl = []
        self.buffer_rr = []

        self.std_fl = 0.0
        self.std_fr = 0.0
        self.std_rl = 0.0
        self.std_rr = 0.0

    def update_from_serial(self, line):
        if line.startswith("RPM_ALL"):
            try:
                parts = line.strip().split(",")
                if len(parts) == 5:
                    self.rpm_fl = float(parts[1])
                    self.rpm_fr = float(parts[2])
                    self.rpm_rl = float(parts[3])
                    self.rpm_rr = float(parts[4])

                    self._update_buffer_and_std(self.rpm_fl, self.buffer_fl, 'fl')
                    self._update_buffer_and_std(self.rpm_fr, self.buffer_fr, 'fr')
                    self._update_buffer_and_std(self.rpm_rl, self.buffer_rl, 'rl')
                    self._update_buffer_and_std(self.rpm_rr, self.buffer_rr, 'rr')
                else:
                    print("[RPMReader] Formato incorrecto:", line)
            except (ValueError, IndexError) as e:
                print("[RPMReader] Error al parsear la línea:", line)
                print("→", e)

    def _update_buffer_and_std(self, value, buffer, wheel):
        buffer.append(value)
        if len(buffer) > self.window_size:
            buffer.pop(0)

        if len(buffer) == self.window_size:
            std = np.std(buffer)
            if wheel == 'fl':
                self.std_fl = self._ema(self.std_fl, std)
            elif wheel == 'fr':
                self.std_fr = self._ema(self.std_fr, std)
            elif wheel == 'rl':
                self.std_rl = self._ema(self.std_rl, std)
            elif wheel == 'rr':
                self.std_rr = self._ema(self.std_rr, std)

    def _ema(self, prev, current):
        return self.alpha * current + (1 - self.alpha) * prev

    def get_all_rpms(self):
        return self.rpm_fl, self.rpm_fr, self.rpm_rl, self.rpm_rr

    def get_all_stds(self):
        return self.std_fl, self.std_fr, self.std_rl, self.std_rr

    def __str__(self):
        return (f"RPMs - FL: {self.rpm_fl:.2f} (σ={self.std_fl:.2f}), "
                f"FR: {self.rpm_fr:.2f} (σ={self.std_fr:.2f}), "
                f"RL: {self.rpm_rl:.2f} (σ={self.std_rl:.2f}), "
                f"RR: {self.rpm_rr:.2f} (σ={self.std_rr:.2f})")



class CoordinatesListener:
    def __init__(self):
        self.rock_coords = ListQueueSimple()
        rospy.Subscriber("obstacle_coordinates", Point, self.callback)

    def callback(self, msg):
        coord = [msg.x, msg.y]  # Use list for compatibility with waypoints
        self.rock_coords.enqueue(coord)  # Or push_front(coord) for newest-first priority

    def get_new_coords(self):
        return self.rock_coords


class SynchronizedData:
    def __init__(self):
        self.lock = Lock()
        self.last_update_time = 0
        self.imu_data = {}
        self.rpm_data = (0, 0, 0, 0)

    def update_imu(self, imu_dict):
        with self.lock:
            self.imu_data = imu_dict
            self.last_update_time = time.time()

    def update_rpm(self, rpm_tuple):
        with self.lock:
            self.rpm_data = rpm_tuple
            self.last_update_time = time.time()

    def get_latest_data(self):
        with self.lock:
            return {
                'imu': self.imu_data,
                'rpm': self.rpm_data,
                'timestamp': self.last_update_time
            }

def initialize_serial(port, baud_rate, timeout):
    """Inicializa la conexión serial y retorna el objeto serial."""
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud_rate,
            timeout=timeout
        )
        print(f"Conectado al puerto {port} a {baud_rate} baudios.")
        time.sleep(2)  # Espera para que Arduino se inicialice
        return ser
    except serial.SerialException as e:
        print(f"Error al conectar al puerto {port}: {e}")
        return None
    
def send_rpm_command(ser, rpm1, rpm2, rpm3, rpm4):
    """Envía una cadena con los RPM de las 4 llantas en el formato M1:rpm1;M2:rpm2;M3:rpm3;M4:rpm4\n."""
    if ser is None or not ser.is_open:
        print("Error: No hay conexión serial activa.")
        return

    # Formatear la cadena
    command = f"M1:{rpm1};M2:{rpm2};M3:{rpm3};M4:{rpm4}\n"
    try:
        # Enviar la cadena codificada
        ser.write(command.encode('utf-8'))
        print(f"Enviado: {command.strip()}")
    except serial.SerialException as e:
        print(f"Error al enviar el comando: {e}")
        
