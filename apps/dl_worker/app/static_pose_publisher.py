"""
Static Pose Publisher — SfM 기반 고정 센서 전역 Pose를 /tf_static으로 발행

Phase 1에서 COLMAP SfM + PnP로 추정된 고정 카메라의 전역 pose를
ROS 2의 /tf_static 토픽으로 발행합니다.

사용법:
  ros2 run dl_worker static_pose_publisher --ros-args \
    -p poses_json_path:=/path/to/sfm_all_poses.json \
    -p world_frame:=world \
    -p sensor_prefix:=sensor_fixed
"""
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class StaticPosePublisher(Node):
    """SfM 추정 고정 카메라 pose → /tf_static 발행 노드"""

    def __init__(self):
        super().__init__('static_pose_publisher')

        # ── Parameters ──────────────────────────────────────
        self.declare_parameter('poses_json_path', '')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('sensor_prefix', 'sensor_fixed')
        # 선택: 특정 카메라 이름만 발행 (콤마 구분, 빈 문자열이면 전체)
        self.declare_parameter('camera_names', '')

        poses_path = self.get_parameter('poses_json_path').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.sensor_prefix = self.get_parameter('sensor_prefix').get_parameter_value().string_value
        camera_names_str = self.get_parameter('camera_names').get_parameter_value().string_value

        self.camera_filter = (
            set(camera_names_str.split(','))
            if camera_names_str
            else None
        )

        # ── Static TF Broadcaster ───────────────────────────
        self._static_broadcaster = StaticTransformBroadcaster(self)

        # ── Load & Publish ───────────────────────────────────
        if not poses_path:
            self.get_logger().error(
                'poses_json_path 파라미터가 비어있습니다. '
                '--ros-args -p poses_json_path:=/path/to/sfm_all_poses.json'
            )
            return

        poses = self._load_poses(poses_path)
        if poses:
            self._publish_static_transforms(poses)

    # ─────────────────────────────────────────────────────────
    # Internal methods
    # ─────────────────────────────────────────────────────────

    def _load_poses(self, path: str) -> dict:
        """sfm_all_poses.json 로드"""
        p = Path(path)
        if not p.exists():
            self.get_logger().error(f'Pose 파일을 찾을 수 없습니다: {path}')
            return {}

        with open(p) as f:
            poses = json.load(f)
        self.get_logger().info(f'Loaded {len(poses)} camera poses from {path}')
        return poses

    def _make_frame_id(self, image_name: str, index: int) -> str:
        """
        이미지 이름으로부터 TF child_frame_id 생성

        예: '20260514_121610.jpg' → 'sensor_fixed_01'
            또는 카메라 이름을 직접 사용: 'cam_collabolab_01'
        """
        return f'{self.sensor_prefix}_{index:02d}'

    def _publish_static_transforms(self, poses: dict):
        """모든 고정 카메라 pose를 /tf_static으로 발행"""
        transforms = []
        now = self.get_clock().now().to_msg()

        for idx, (image_name, pose_data) in enumerate(sorted(poses.items())):
            # 카메라 필터 적용
            if self.camera_filter and image_name not in self.camera_filter:
                continue

            quat = pose_data.get('quaternion_wxyz', [1.0, 0.0, 0.0, 0.0])
            pos = pose_data.get('position_xyz', [0.0, 0.0, 0.0])

            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.world_frame
            t.child_frame_id = self._make_frame_id(image_name, idx)

            # Position (translation)
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])

            # Orientation (quaternion: ROS uses x, y, z, w; COLMAP uses w, x, y, z)
            t.transform.rotation.w = float(quat[0])
            t.transform.rotation.x = float(quat[1])
            t.transform.rotation.y = float(quat[2])
            t.transform.rotation.z = float(quat[3])

            transforms.append(t)

            self.get_logger().info(
                f'  {t.child_frame_id} ← {image_name}: '
                f'pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), '
                f'obs={pose_data.get("num_observations", "?")}'
            )

        if transforms:
            self._static_broadcaster.sendTransform(transforms)
            self.get_logger().info(
                f'Published {len(transforms)} static transforms '
                f'({self.world_frame} → {self.sensor_prefix}_*) on /tf_static'
            )
        else:
            self.get_logger().warn('No transforms to publish!')


def main(args=None):
    rclpy.init(args=args)
    node = StaticPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down StaticPosePublisher.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
