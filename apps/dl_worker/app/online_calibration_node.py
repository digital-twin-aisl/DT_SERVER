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
Online Calibration Node — ROS 2 센서 간 Online Extrinsic Calibration 노드

Phase 3의 핵심 노드로, 다중 센서의 이미지를 수신하여
Overlapping FOV를 자동 감지하고, 센서 쌍의 상대 변환을 추정합니다.

Subscribers:
  /edge/+/image        → sensor_msgs/Image (다중 센서 이미지)
  /sensor/image/+      → sensor_msgs/Image (대안 토픽 패턴)

Publishers:
  /sensor/extrinsic    → geometry_msgs/TransformStamped (상대 변환)
  /calibration/status  → std_msgs/String (상태 정보)

Parameters:
  - world_frame: TF 기준 프레임 (default: 'world')
  - device: GPU 디바이스 (default: 'cuda')
  - max_keypoints: SuperPoint 최대 키포인트 (default: 2048)
  - calibration_rate_hz: 캘리브레이션 주기 (default: 1.0)
  - min_matches: 최소 매치 수 (default: 30)
  - sensor_topics: 센서 이미지 토픽 목록 (JSON array)
  - sfm_model_path: SfM 맵 경로 (scale resolution용)

사용법:
  ros2 run dl_worker online_calibration_node --ros-args \\
    -p device:=cuda \\
    -p calibration_rate_hz:=1.0 \\
    -p sensor_topics:='["/edge/cam_01/image", "/edge/cam_02/image"]'
"""
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import threading
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from scripts.phase3_online_calibration.online_calibration_manager import (
    OnlineCalibrationManager,
)

# Optional imports
try:
    from scripts.phase2_pose_graph.pose_graph_manager import PoseGraphManager
    POSE_GRAPH_AVAILABLE = True
except ImportError:
    POSE_GRAPH_AVAILABLE = False

try:
    from scripts.phase1_sfm.sfm_map import SfMMap
    SFM_AVAILABLE = True
except ImportError:
    SFM_AVAILABLE = False


class OnlineCalibrationNode(Node):
    """
    ROS 2 Online Extrinsic Calibration Node

    다중 센서의 이미지를 수신하여 overlapping FOV 감지 및
    상대 pose 추정을 자동으로 수행합니다.
    """

    def __init__(self):
        super().__init__('online_calibration_node')

        # ── Parameters ──────────────────────────────────────
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('max_keypoints', 2048)
        self.declare_parameter('calibration_rate_hz', 1.0)
        self.declare_parameter('min_matches', 30)
        self.declare_parameter('min_match_ratio', 0.05)
        self.declare_parameter('min_inlier_ratio', 0.5)
        self.declare_parameter('ransac_threshold', 1.0)
        self.declare_parameter('min_pose_inliers', 20)
        self.declare_parameter('sensor_topics', '[]')  # JSON array
        self.declare_parameter('sfm_model_path', '')
        self.declare_parameter('default_fx', 800.0)
        self.declare_parameter('default_fy', 800.0)
        self.declare_parameter('default_cx', 320.0)
        self.declare_parameter('default_cy', 240.0)

        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value
        max_kp = self.get_parameter('max_keypoints').get_parameter_value().integer_value
        cal_rate = self.get_parameter('calibration_rate_hz').get_parameter_value().double_value
        min_matches = self.get_parameter('min_matches').get_parameter_value().integer_value
        min_match_ratio = self.get_parameter('min_match_ratio').get_parameter_value().double_value
        min_inlier_ratio = self.get_parameter('min_inlier_ratio').get_parameter_value().double_value
        ransac_threshold = self.get_parameter('ransac_threshold').get_parameter_value().double_value
        min_pose_inliers = self.get_parameter('min_pose_inliers').get_parameter_value().integer_value
        sensor_topics_str = self.get_parameter('sensor_topics').get_parameter_value().string_value
        sfm_path = self.get_parameter('sfm_model_path').get_parameter_value().string_value

        # Default intrinsic
        self._default_fx = self.get_parameter('default_fx').get_parameter_value().double_value
        self._default_fy = self.get_parameter('default_fy').get_parameter_value().double_value
        self._default_cx = self.get_parameter('default_cx').get_parameter_value().double_value
        self._default_cy = self.get_parameter('default_cy').get_parameter_value().double_value

        # ── Parse sensor topics ──────────────────────────────
        try:
            self._sensor_topics = json.loads(sensor_topics_str) if sensor_topics_str else []
        except json.JSONDecodeError:
            self._sensor_topics = []
            self.get_logger().warn(f'Invalid sensor_topics JSON: {sensor_topics_str}')

        # ── Load SfM map (optional) ──────────────────────────
        sfm_map = None
        if sfm_path and SFM_AVAILABLE:
            try:
                sfm_map = SfMMap(sfm_path)
                self.get_logger().info(f'Loaded SfM map from {sfm_path}')
            except Exception as e:
                self.get_logger().warn(f'Failed to load SfM map: {e}')

        # ── Initialize calibration manager ───────────────────
        self.calibration_manager = OnlineCalibrationManager(
            pose_graph=None,  # PoseGraph는 별도로 주입
            sfm_map=sfm_map,
            device=device,
            max_keypoints=max_kp,
            min_matches=min_matches,
            min_match_ratio=min_match_ratio,
            min_inlier_ratio=min_inlier_ratio,
            ransac_threshold=ransac_threshold,
            min_pose_inliers=min_pose_inliers,
        )

        # ── Image buffer (센서별 최신 이미지) ────────────────
        self._image_buffer = {}      # sensor_id → np.ndarray (BGR)
        self._intrinsic_buffer = {}  # sensor_id → 3x3 intrinsic
        self._buffer_lock = threading.Lock()
        self._cv_bridge = CvBridge()

        # ── Subscribers ──────────────────────────────────────
        reliable_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE)

        self._image_subs = []
        for topic in self._sensor_topics:
            sub = self.create_subscription(
                Image, topic,
                self._make_image_callback(topic),
                reliable_qos,
            )
            self._image_subs.append(sub)
            self.get_logger().info(f'Subscribing to sensor topic: {topic}')

        # CameraInfo subscribers (intrinsic 수신)
        self._camera_info_subs = []
        for topic in self._sensor_topics:
            # Convention: /edge/cam_01/image → /edge/cam_01/camera_info
            info_topic = topic.rsplit('/', 1)[0] + '/camera_info'
            sub = self.create_subscription(
                CameraInfo, info_topic,
                self._make_camera_info_callback(info_topic),
                QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE),
            )
            self._camera_info_subs.append(sub)

        # ── Publishers ───────────────────────────────────────
        self.extrinsic_pub = self.create_publisher(
            TransformStamped,
            '/sensor/extrinsic',
            QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE),
        )

        self.status_pub = self.create_publisher(
            String,
            '/calibration/status',
            QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE),
        )

        # ── Timer: 주기적 캘리브레이션 ────────────────────────
        if cal_rate > 0:
            self.cal_timer = self.create_timer(1.0 / cal_rate, self._calibration_tick)

        self.get_logger().info(
            f"OnlineCalibrationNode started: "
            f"device={device}, rate={cal_rate}Hz, "
            f"{len(self._sensor_topics)} sensor topics"
        )

    def _make_image_callback(self, topic: str):
        """토픽별 이미지 콜백 생성"""
        def callback(msg: Image):
            sensor_id = msg.header.frame_id or self._topic_to_sensor_id(topic)
            try:
                cv_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                with self._buffer_lock:
                    self._image_buffer[sensor_id] = cv_image
            except Exception as e:
                self.get_logger().error(f'Image conversion error ({sensor_id}): {e}')
        return callback

    def _make_camera_info_callback(self, topic: str):
        """CameraInfo 콜백 생성"""
        def callback(msg: CameraInfo):
            sensor_id = msg.header.frame_id or self._topic_to_sensor_id(topic)
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            with self._buffer_lock:
                self._intrinsic_buffer[sensor_id] = K
        return callback

    @staticmethod
    def _topic_to_sensor_id(topic: str) -> str:
        """토픽 이름 → 센서 ID 추출 (e.g. /edge/cam_01/image → cam_01)"""
        parts = topic.strip('/').split('/')
        if len(parts) >= 2:
            return parts[-2]  # 'cam_01'
        return parts[-1]

    def _get_default_intrinsic(self) -> np.ndarray:
        """기본 intrinsic matrix 반환"""
        return np.array([
            [self._default_fx, 0, self._default_cx],
            [0, self._default_fy, self._default_cy],
            [0, 0, 1],
        ], dtype=np.float64)

    def _calibration_tick(self):
        """주기적 캘리브레이션 수행"""
        with self._buffer_lock:
            images = dict(self._image_buffer)
            intrinsics = dict(self._intrinsic_buffer)

        if len(images) < 2:
            return  # 최소 2개 센서 필요

        # Intrinsic이 없는 센서에는 default 적용
        default_K = self._get_default_intrinsic()
        for sensor_id in images:
            if sensor_id not in intrinsics:
                intrinsics[sensor_id] = default_K

        # 캘리브레이션 실행
        result = self.calibration_manager.calibrate(
            images=images,
            camera_intrinsics=intrinsics,
        )

        # 결과 발행
        if result.pose_estimates:
            self._publish_extrinsics()
            self._publish_status(result)

    def _publish_extrinsics(self):
        """현재 캘리브레이션된 extrinsic을 /sensor/extrinsic으로 발행"""
        from scipy.spatial.transform import Rotation

        now = self.get_clock().now().to_msg()

        for pair_data in self.calibration_manager.to_transform_stamped_list():
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now
            tf_msg.header.frame_id = pair_data['parent_frame']
            tf_msg.child_frame_id = pair_data['child_frame']

            t = pair_data['translation']
            q = pair_data['rotation_xyzw']

            tf_msg.transform.translation.x = float(t[0])
            tf_msg.transform.translation.y = float(t[1])
            tf_msg.transform.translation.z = float(t[2])
            tf_msg.transform.rotation.x = float(q[0])
            tf_msg.transform.rotation.y = float(q[1])
            tf_msg.transform.rotation.z = float(q[2])
            tf_msg.transform.rotation.w = float(q[3])

            self.extrinsic_pub.publish(tf_msg)

    def _publish_status(self, result):
        """캘리브레이션 상태를 JSON으로 발행"""
        stats = self.calibration_manager.get_stats()
        status = {
            'cycle': stats['total_cycles'],
            'overlapping_pairs': len(result.overlapping_pairs),
            'pose_estimates': len(result.pose_estimates),
            'factors_injected': result.factors_injected,
            'elapsed_ms': round(result.elapsed_seconds * 1000, 1),
            'total_tracked_pairs': stats['num_tracked_pairs'],
        }

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OnlineCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('OnlineCalibrationNode shutting down.')
    finally:
        stats = node.calibration_manager.get_stats()
        node.get_logger().info(f'Final stats: {stats}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
