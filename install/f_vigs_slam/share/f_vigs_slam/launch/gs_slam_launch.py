from launch import LaunchDescription
from launch_ros.actions import Node

# Definimos el archivo para lanzar el nodo a ROS2

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='f_vigs_slam',
            executable='gs_slam_node',
            output='screen'
        )
    ])
