# controller.py
import numpy as np
from utils import ListQueueSimple 
import rospy

def angulo_ackermann(delta, v2):
    """
    Calcula el ángulo de Ackermann ajustado y la velocidad interior.
    """
    distancia1 = np.tan(delta) * 28.15
    d2 = distancia1 + 89
    beta = np.atan(28.15 / d2)
    b = (np.pi / 2) - beta
    a = (np.pi / 2) - delta
    v1 = ((np.tan(a) * 28.15) / (89 + (np.tan(a) * 28.15))) * v2
    return b, v1

def find_look_ahead_point(x, y, waypoints, idx_current, Ld,piedras =None):
    """
    Busca el siguiente punto de seguimiento (look-ahead point) a lo largo de la trayectoria.
    """
    
    N = waypoints.shape[0]
    lx = waypoints[-1, 0]
    ly = waypoints[-1, 1]
    idx_next = idx_current
    robot_pos = np.array([x, y])

    # Recorre los segmentos de la trayectoria
    while idx_next < N - 1:
        seg_start = waypoints[idx_next]
        seg_end = waypoints[idx_next + 1]
        seg_vec = seg_end - seg_start
        seg_len = np.linalg.norm(seg_vec)
        to_start = robot_pos - seg_start
        proj = np.dot(to_start, seg_vec) / seg_len
        remain = seg_len - proj

        if remain < 0:
            idx_next += 1
            continue

        s = Ld if remain >= Ld else remain
        param = proj + s
        param_frac = max(0, min(1, param / seg_len))
        look_pt = seg_start + param_frac * seg_vec
        lx, ly = look_pt

        if s >= remain:
            idx_next += 1
        else:
            break

    return lx, ly, idx_next


def generar_ruta_prioritaria(piedras_lista, use_push_front=False):
    """
    Get the next waypoint, prioritizing stones over predefined waypoints.

    Args:
        piedras_lista: ListQueueSimple with stone coordinates.
        use_push_front: If True, assumes stones are added with push_front.

    Returns:
        list: Next waypoint ([x, y]) or None if no points remain.
    """
    # Persistent waypoints queue
    if not hasattr(generar_ruta_prioritaria, 'puntos_establecidos'):
        generar_ruta_prioritaria.puntos_establecidos = ListQueueSimple()
        waypoints_array = np.array([
            [1.295, 1.5], [1.295, 6.5], [3.777, 6.5],
            [3.777, 1.5], [6.475, 1.5], [6.475, 6.5],
            [9.065, 6.5], [9.065, 1.5], [1.295, 1.5]
        ])
        for punto in waypoints_array:
            generar_ruta_prioritaria.puntos_establecidos.enqueue(punto.tolist())

    # Get next point
    try:
        if not piedras_lista.isempty():
            punto = piedras_lista.dequeue()
            rospy.loginfo(f"Prioritizing detected stone at {punto}")
        elif not generar_ruta_prioritaria.puntos_establecidos.isempty():
            punto = generar_ruta_prioritaria.puntos_establecidos.dequeue()
            rospy.loginfo(f"Following predefined waypoint at {punto}")
        else:
            rospy.loginfo("No points remain")
            return None

        # Validate point
        if not isinstance(punto, (list, np.ndarray)) or len(punto) != 2:
            rospy.logwarn(f"Invalid point: {punto}")
            return None

        return punto
    except Exception as e:
        rospy.logerr(f"Error processing point: {e}")
        return None
    
def find_stopping_point(rock_pixel_x, rock_distance, current_x, current_y, current_theta):
    # --- Parámetros de entrada ---
    FOV_deg     = 60      # Campo de visión horizontal de la cámara (º)
    image_width = 640     # Ancho de la imagen (px)
    r_stop      = 1.0     # Distancia de seguridad antes de la roca (m)

    # --- Paso 1: Pixel --> ángulo en radianes ---
    deg_per_px   = FOV_deg / image_width
    center_px    = image_width / 2
    pixel_offset = rock_pixel_x - center_px
    angle_rad    = np.deg2rad(pixel_offset * deg_per_px)

    # --- Paso 2: Posición de la roca en marco robot (x hacia adelante, y lateral) ---
    x_rock = rock_distance * np.cos(angle_rad)
    y_rock = rock_distance * np.sin(angle_rad)

    # --- Paso 3: Punto de paro en marco robot ---
    dist     = np.hypot(x_rock, y_rock)
    ux, uy   = x_rock/dist, y_rock/dist
    x_stop_r = x_rock - r_stop * ux
    y_stop_r = y_rock - r_stop * uy

    # --- Paso 4: Transformación a coordenadas globales ---
    # current_theta en radianes; 0 = eje X global
    ct = np.cos(current_theta)
    st = np.sin(current_theta)

    x_stop_global = current_x + ( x_stop_r * ct - y_stop_r * st )
    y_stop_global = current_y + ( x_stop_r * st + y_stop_r * ct )

    return [x_stop_global, y_stop_global]

def robot_stop(stopping_point, current_x, current_y, threshold=0.3):
    stop_x, stop_y = stopping_point
    distance = np.hypot(stop_x - current_x, stop_y - current_y)
    return distance <= threshold