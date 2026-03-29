from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    home_dir = os.environ['HOME']
    base_path = os.path.join(home_dir, 'ros2_ws', 'src', 'terrain_mapping_drone_control', 'terrain_mapping_drone_control')

    # FastDDS profile to fix payload size mismatch
    pkg_share = get_package_share_directory('terrain_mapping_drone_control')
    fastdds_profile = os.path.join(pkg_share, 'config', 'fastdds_profile.xml')

    return LaunchDescription([
        # Run the terrain mission controller with FastDDS fix
        ExecuteProcess(
            cmd=['python3', os.path.join(base_path, 'terrain_mission.py')],
            output='screen',
            additional_env={'FASTRTPS_DEFAULT_PROFILES_FILE': fastdds_profile},
        )
    ])
