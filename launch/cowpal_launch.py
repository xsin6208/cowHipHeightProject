import os
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    # Launch heightObtainor and calibrator immediately
    height_obtainor_node = Node(
        package='cowpal',
        executable='heightObtainor.py',
        name='HeightObtainor',
        output='screen'
    )

    calibrator_node = Node(
        package='cowpal',
        executable='calibration.py',
        name='Calibrator',
        output='screen'
    )

    # Launch websiteWriter immediately after the first two
    website_writer_node = Node(
        package='cowpal',
        executable='websiteWriter.py',
        name='website_writer',
        output='screen'
    )

    # Launch manager.py after a delay (e.g., 5 seconds)
    manager_node = TimerAction(
        period=10.0,  # seconds
        actions=[Node(
            package='cowpal',
            executable='manager.py',
            name='manager',
            output='screen'
        )]
    )

    return LaunchDescription([
        height_obtainor_node,
        calibrator_node,
        website_writer_node,
        manager_node
    ])
