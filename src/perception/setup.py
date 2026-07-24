from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mjw99',
    description='Perception Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_processing = perception.camera_processing:main',
            'lidar_processing = perception.lidar_processing:main',
            'radar_processing = perception.radar_processing:main',
            'sensor_fusion = perception.sensor_fusion_node:main',

        ],
    },
)
