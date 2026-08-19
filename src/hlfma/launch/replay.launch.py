"""
rosbag 의 /gt_state 만 재생해서 VTD 없이 나머지 노드를 돌린다.

  ros2 launch hlfma replay.launch.py bag:=bags/drive

vtd_bridge 는 띄우지 않는다(소켓 불필요). 대신 bag 이 /gt_state 를 뿌리고
perception → planner → control 콜백 체인이 그대로 돈다.
녹화본의 /world_state /decision /cmd 는 재생하지 않는다 — 새로 계산한 결과와
섞이면 회귀 비교가 무의미해지기 때문이다.

  rate:=2.0     배속
  record:=true  재계산 결과를 새 bag 으로 저장 (원본과 비교용)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('hlfma')
    default_params = os.path.join(share, 'config', 'params.yaml')

    bag = LaunchConfiguration('bag')
    rate = LaunchConfiguration('rate')
    graph = LaunchConfiguration('graph')
    route = LaunchConfiguration('route')
    params = LaunchConfiguration('params')
    record = LaunchConfiguration('record')
    out_bag = LaunchConfiguration('out_bag')

    common = [params, {'graph_path': graph, 'route_path': route}]

    return LaunchDescription([
        DeclareLaunchArgument('bag'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('graph', default_value='data/lane_graph.pkl'),
        DeclareLaunchArgument('route', default_value='data/route.pkl'),
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('record', default_value='false'),
        DeclareLaunchArgument('out_bag', default_value='bags/replay_out'),

        Node(package='hlfma', executable='perception', name='perception',
             output='screen', parameters=common),
        Node(package='hlfma', executable='planner', name='planner',
             output='screen', parameters=common),
        Node(package='hlfma', executable='control', name='control',
             output='screen', parameters=common),
        Node(package='hlfma', executable='logger', name='logger',
             output='screen', parameters=common),

        # /gt_state 만 재생한다
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag, '--rate', rate, '--topics', '/gt_state'],
            output='screen',
        ),
        ExecuteProcess(
            condition=IfCondition(record),
            cmd=['ros2', 'bag', 'record', '-o', out_bag,
                 '/gt_state', '/world_state', '/decision', '/cmd'],
            output='screen',
        ),
    ])
