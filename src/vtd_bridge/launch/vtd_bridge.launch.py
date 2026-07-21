from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vtd_bridge',
            executable='rdb_receiver',
            name='rdb_receiver',
            output='screen'
        )
    ])
