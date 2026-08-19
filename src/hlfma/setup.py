import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'hlfma'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mjw',
    maintainer_email='kaai37707@gmail.com',
    description='HL FMA 2026 자율주행 컨트롤러 (룰베이스)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vtd_bridge = hlfma.nodes.vtd_bridge_node:main',
            'perception = hlfma.nodes.perception_node:main',
            'planner    = hlfma.nodes.planner_node:main',
            'control    = hlfma.nodes.control_node:main',
            'logger     = hlfma.nodes.logger_node:main',
            'drive      = hlfma.nodes.single_process:main',
        ],
    },
)
