from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camara_pkg',
            executable='depthGetter',
            name='profundidad',
            output='screen'
        ),
        Node(
            package='camara_pkg',
            executable='rgbImageGetter',
            name='color',
            output='screen'
        ),
        Node(
            package='lidar_pkg',
            executable='lidarGetter',
            name='LiDAR',
            output='screen'
        ),
        Node(
            package='autonomy_pkg',
            executable='pathPlanner',
            name='autonomia',
            output='screen'
        ),
        Node(
            package='hector_slam_pkg',
            executable='mapGenerator',
            name='mapa',
            output='screen'
        ),
        Node(
            package='arm_pkg',
            executable='armOperator',
            name='brazoO',
            output='screen'
        ),
        Node(
            package='arm_pkg',
            executable='armSetter',
            name='brazoS',
            output='screen'
        ),
    ])