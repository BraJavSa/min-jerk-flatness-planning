#!/usr/bin/env python3
import sys
from pathlib import Path
from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE_NAME = 'min-jerk-flatness-planning'
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / 'output')

def generate_launch_description():
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value=DEFAULT_OUTPUT_DIR,
        description='Directory for CSV/JSON reference and tracking results'
    )
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/wamv/sensors/position/ground_truth_odometry',
        description='Vehicle ground truth odometry topic'
    )
    planner_binary_arg = DeclareLaunchArgument(
        'planner_binary',
        default_value='',
        description='Path to C planner binary (leave empty for auto-detection)'
    )
    rate_arg = DeclareLaunchArgument(
        'rate', default_value='30.0', description='NMPC control and recording frequency (Hz)'
    )

    planner_node = Node(
        package=PACKAGE_NAME,
        executable='pseudo_flat_planner',
        name='pseudo_flat_planner',
        output='screen',
        parameters=[{
            'output_dir': LaunchConfiguration('output_dir'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'planner_binary': LaunchConfiguration('planner_binary'),
        }],
    )

    mpc_node = Node(
        package=PACKAGE_NAME,
        executable='pseudo_flat_mpc_node',
        name='pseudo_flat_mpc_node',
        output='screen',
        parameters=[{
            'output_dir': LaunchConfiguration('output_dir'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'rate': LaunchConfiguration('rate'),
        }],
    )

    recorder_node = Node(
        package=PACKAGE_NAME,
        executable='pseudo_flat_recorder',
        name='pseudo_flat_recorder',
        output='screen',
        parameters=[{
            'output_dir': LaunchConfiguration('output_dir'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'rate': LaunchConfiguration('rate'),
        }],
    )

    return LaunchDescription([
        output_dir_arg,
        odom_topic_arg,
        planner_binary_arg,
        rate_arg,
        planner_node,
        mpc_node,
        recorder_node,
    ])

if __name__ == '__main__':
    ls = LaunchService(argv=sys.argv[1:])
    ls.include_launch_description(generate_launch_description())
    sys.exit(ls.run())
