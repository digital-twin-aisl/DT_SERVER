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
Phase 3 Launch File — 다중 센서 Online Targetless Extrinsic Calibration

Phase 2 시스템 위에 online_calibration_node를 추가하여, FOV가 겹치는
센서 쌍을 자동 감지하고 /sensor/extrinsic 토픽으로 상대 변환을 발행합니다.
pose_graph_node가 이 토픽을 구독하여 inter-sensor BetweenFactor를 주입합니다.

기동 순서:
  1. static_pose_publisher (Phase 1 고정 센서 /tf_static)
  2. pose_graph_node (GTSAM iSAM2 중앙 최적화)
  3. relocalization_node (HLoc Visual Relocalization)
  4. vio_bridge_node (VIO → Odometry)
  5. online_calibration_node (Phase 3 core)
  6. dl_inference_node (추론)

사용법:
  ros2 launch dl_worker phase3_launch.py \
    sfm_model_path:=/data/sfm_output/sparse_txt \
    db_image_dir:=/data/collab_images \
    poses_json_path:=/data/sfm_all_poses.json \
    vio_mode:=stub \
    sensor_topics:='["/edge/cam_01/image","/edge/cam_02/image"]'
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
        # Phase 3 전용 인자
        DeclareLaunchArgument('sensor_topics', default_value='[]'),
        DeclareLaunchArgument('calibration_rate_hz', default_value='1.0'),
        DeclareLaunchArgument('min_matches', default_value='30'),
        DeclareLaunchArgument('min_match_ratio', default_value='0.05'),
        DeclareLaunchArgument('min_inlier_ratio', default_value='0.5'),
        DeclareLaunchArgument('ransac_threshold', default_value='1.0'),
        DeclareLaunchArgument('min_pose_inliers', default_value='20'),
        DeclareLaunchArgument('max_keypoints', default_value='2048'),

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
                'max_keypoints': LaunchConfiguration('max_keypoints'),
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

        # ── 5. Online Calibration Node (Phase 3 core) ──────
        Node(
            package='dl_worker',
            executable='online_calibration_node',
            name='online_calibration_node',
            parameters=[{
                'world_frame': LaunchConfiguration('world_frame'),
                'device': LaunchConfiguration('device'),
                'max_keypoints': LaunchConfiguration('max_keypoints'),
                'calibration_rate_hz': LaunchConfiguration('calibration_rate_hz'),
                'min_matches': LaunchConfiguration('min_matches'),
                'min_match_ratio': LaunchConfiguration('min_match_ratio'),
                'min_inlier_ratio': LaunchConfiguration('min_inlier_ratio'),
                'ransac_threshold': LaunchConfiguration('ransac_threshold'),
                'min_pose_inliers': LaunchConfiguration('min_pose_inliers'),
                'sensor_topics': LaunchConfiguration('sensor_topics'),
                'sfm_model_path': LaunchConfiguration('sfm_model_path'),
            }],
            output='screen',
        ),

        # ── 6. DL Inference Node ───────────────────────────
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
