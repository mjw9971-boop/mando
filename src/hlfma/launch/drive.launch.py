"""
주행 실행.

  ros2 launch hlfma drive.launch.py graph:=data/lane_graph.pkl route:=data/route.pkl

인자
  graph, route         지도/경로 pkl 경로
  host, port           VTD 주소
  params               파라미터 파일 (기본: 패키지의 config/params.yaml)
  use_single_process   true(기본)= 한 프로세스에 4노드 / false = 노드별 프로세스
  record               true 면 전 토픽 rosbag 녹화
  bag_path             녹화 경로
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

NODES = [('vtd_bridge', 'vtd_bridge'), ('perception', 'perception'),
         ('planner', 'planner'), ('control', 'control'), ('logger', 'logger')]


def generate_launch_description():
    share = get_package_share_directory('hlfma')
    default_params = os.path.join(share, 'config', 'params.yaml')

    graph = LaunchConfiguration('graph')
    route = LaunchConfiguration('route')
    host = LaunchConfiguration('host')
    port = LaunchConfiguration('port')
    params = LaunchConfiguration('params')
    single = LaunchConfiguration('use_single_process')
    record = LaunchConfiguration('record')
    bag_path = LaunchConfiguration('bag_path')

    # 파라미터 파일 위에 launch 인자를 덮어쓴다
    debug_speed = LaunchConfiguration('debug_speed')
    overrides = {
        'graph_path': graph, 'route_path': route,
        'comm.host': host, 'comm.port': port,
        # debug_speed > 0 이면 상수속도 모드. 기본 0 = 끔.
        # (기본 주행은 제한속도·곡률·정지선만으로 속도를 정한다)
        'debug.const_speed_kph': ParameterValue(debug_speed, value_type=float),
        'debug.enabled': ParameterValue(
            PythonExpression(['float("', debug_speed, '") > 0']), value_type=bool),
    }
    common = [params, overrides]

    return LaunchDescription([
        DeclareLaunchArgument('graph', default_value='data/lane_graph.pkl'),
        DeclareLaunchArgument('route', default_value='data/route.pkl'),
        DeclareLaunchArgument('host', default_value='127.0.0.1'),
        DeclareLaunchArgument('port', default_value='9910'),
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('use_single_process', default_value='true',
                              description='true = 4노드를 한 프로세스에'),
        DeclareLaunchArgument('record', default_value='false',
                              description='true 면 전 토픽 rosbag 녹화'),
        DeclareLaunchArgument('bag_path', default_value='bags/drive'),
        DeclareLaunchArgument('debug_speed', default_value='0.0',
                              description='[km/h] >0 이면 상수속도 모드(연동 확인용). '
                                          '0 = 제한속도·곡률·정지선으로만 주행'),

        # 단일 프로세스 (기본)
        # name= 을 주면 __node 리맵이 걸려 프로세스 안 4개 노드가 모두 같은 이름이
        # 된다(rosout 경고 + 노드별 파라미터 불가). 각자 이름을 쓰게 둔다.
        Node(package='hlfma', executable='drive',
             output='screen', parameters=common,
             condition=IfCondition(single)),

        # 분리 실행
        GroupAction(
            condition=UnlessCondition(single),
            actions=[
                Node(package='hlfma', executable=exe, name=name,
                     output='screen', parameters=common)
                for exe, name in NODES
            ],
        ),

        ExecuteProcess(
            condition=IfCondition(record),
            cmd=['ros2', 'bag', 'record', '-o', bag_path,
                 '/gt_state', '/world_state', '/decision', '/cmd'],
            output='screen',
        ),
    ])
