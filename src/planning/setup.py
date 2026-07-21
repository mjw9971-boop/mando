from setuptools import find_packages, setup

package_name = 'planning'

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
    description='Planning Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'path_planner = planning.path_planner:main',
        ],
    },
)
