# This file is part of DT_SERVER.
# 
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
# 
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

"""
Phase 2 Launch File — 이동형 센서 지원 전체 시스템 기동

기동 순서:
  1. static_pose_publisher (Phase 1 고정 센서 /tf_static)
  2. pose_graph_node (GTSAM iSAM2 중앙 최적화)
  3. relocalization_node (HLoc Visual Relocalization)
  4. vio_bridge_node (VIO → Odometry)
  5. dl_inference_node (추론)

사용법:
  ros2 launch dl_worker phase2_launch.py \
    sfm_model_path:=/data/sfm_output/sparse_txt \
    db_image_dir:=/data/collab_images \
    poses_json_path:=/data/sfm_all_poses.json \
    vio_mode:=stub
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ── Launch Arguments ────────────────────────────────
        DeclareLaunchArgument('world_frame', default_value='world'),
        DeclareLaunchArgument('poses_json_path', default_value=''),
        DeclareLaunchArgument('sfm_model_path', default_value=''),
        DeclareLaunchArgument('db_image_dir', default_value=''),
        DeclareLaunchArgument('vio_mode', default_value='stub'),
        DeclareLaunchArgument('sensor_frame', default_value='robot_01/base_link'),
        DeclareLaunchArgument('device', default_value='cuda'),

        # ── 1. Static Pose Publisher (Phase 1) ──────────────
        Node(
            package='dl_worker',
            executable='static_pose_publisher',
            name='static_pose_publisher',
            parameters=[{
                'poses_json_path': LaunchConfiguration('poses_json_path'),
                'world_frame': LaunchConfiguration('world_frame'),
                'sensor_prefix': 'sensor_fixed',
            }],
            output='screen',
        ),

        # ── 2. Pose Graph Node (Phase 2 core) ──────────────
        Node(
            package='dl_worker',
            executable='pose_graph_node',
            name='pose_graph_node',
            parameters=[{
                'world_frame': LaunchConfiguration('world_frame'),
                'update_rate_hz': 10.0,
                'fixed_poses_json': LaunchConfiguration('poses_json_path'),
                'relinearize_threshold': 0.1,
                'relinearize_skip': 10,
            }],
            output='screen',
        ),

        # ── 3. Relocalization Node ─────────────────────────
        Node(
            package='dl_worker',
            executable='relocalization_node',
            name='relocalization_node',
            parameters=[{
                'sfm_model_path': LaunchConfiguration('sfm_model_path'),
                'db_image_dir': LaunchConfiguration('db_image_dir'),
                'device': LaunchConfiguration('device'),
                'max_keypoints': 2048,
                'world_frame': LaunchConfiguration('world_frame'),
                'ransac_threshold': 12.0,
                'min_inliers': 15,
                'top_k_db': 10,
            }],
            output='screen',
        ),

        # ── 4. VIO Bridge Node ─────────────────────────────
        Node(
            package='dl_worker',
            executable='vio_bridge_node',
            name='vio_bridge_node',
            parameters=[{
                'vio_mode': LaunchConfiguration('vio_mode'),
                'sensor_frame': LaunchConfiguration('sensor_frame'),
                'world_frame': LaunchConfiguration('world_frame'),
                'odom_topic': '/robot/odom',
                'stub_velocity': 0.1,
                'stub_angular_vel': 0.05,
                'stub_rate_hz': 10.0,
            }],
            output='screen',
        ),

        # ── 5. DL Inference Node ───────────────────────────
        Node(
            package='dl_worker',
            executable='dl_inference_node',
            name='dl_inference_node',
            parameters=[{
                'world_frame': LaunchConfiguration('world_frame'),
            }],
            output='screen',
        ),
    ])
