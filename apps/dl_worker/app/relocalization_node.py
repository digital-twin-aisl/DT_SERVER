"""
Relocalization Node — HLoc Visual Relocalization ROS 2 Service

Phase 1의 HLocLocalizer (SuperPoint+LightGlue+PnP)를
ROS 2 Service로 래핑하여 이동형 센서의 전역 pose를 제공합니다.

Service:
  /relocalize  (sensor_msgs/Image request → 
                geometry_msgs/PoseWithCovarianceStamped response)

Publisher:
  /sensor/relocalized_pose → PoseWithCovarianceStamped
    (relocalization 성공 시 자동 발행 → pose_graph_node가 수신)

사용법:
  ros2 run dl_worker relocalization_node --ros-args \
    -p sfm_model_path:=/path/to/sparse_txt \
    -p db_image_dir:=/path/to/colmap_images \
    -p device:=cuda \
    -p max_keypoints:=2048
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Trigger
import cv2
from cv_bridge import CvBridge
import tempfile
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

logger = logging.getLogger(__name__)

# Lazy import for GPU-heavy modules
_hloc_localizer = None
_sfm_map_class = None


def _get_hloc_modules():
    global _hloc_localizer, _sfm_map_class
    if _hloc_localizer is None:
        from scripts.phase1_sfm.hloc_localizer import HLocLocalizer
        from scripts.phase1_sfm.sfm_map import SfMMap
        _hloc_localizer = HLocLocalizer
        _sfm_map_class = SfMMap
    return _hloc_localizer, _sfm_map_class


class RelocalizationNode(Node):
    """
    HLoc 기반 Visual Relocalization Service Node

    이동형 센서가 이미지를 보내면 SfM 맵 대비 전역 pose를 추정하여
    PoseWithCovarianceStamped로 발행합니다.
    """

    def __init__(self):
        super().__init__('relocalization_node')

        # ── Parameters ──────────────────────────────────────
        self.declare_parameter('sfm_model_path', '')
        self.declare_parameter('db_image_dir', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('max_keypoints', 2048)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('ransac_threshold', 12.0)
        self.declare_parameter('min_inliers', 15)
        self.declare_parameter('top_k_db', 10)

        sfm_path = self.get_parameter('sfm_model_path').get_parameter_value().string_value
        self.db_image_dir = self.get_parameter('db_image_dir').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value
        max_kp = self.get_parameter('max_keypoints').get_parameter_value().integer_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.ransac_thresh = self.get_parameter('ransac_threshold').get_parameter_value().double_value
        self.min_inliers = self.get_parameter('min_inliers').get_parameter_value().integer_value
        self.top_k = self.get_parameter('top_k_db').get_parameter_value().integer_value

        # ── Initialize HLoc localizer ───────────────────────
        self.localizer = None
        self.cv_bridge = CvBridge()

        if sfm_path and self.db_image_dir:
            try:
                HLocLocalizer, SfMMap = _get_hloc_modules()
                sfm_map = SfMMap(sfm_path)
                self.localizer = HLocLocalizer(
                    sfm_map, device=device, max_keypoints=max_kp
                )
                # Pre-cache DB features
                self.localizer.build_db_features(self.db_image_dir)
                self.get_logger().info(
                    f'HLoc localizer initialized: {sfm_path}, '
                    f'device={device}, max_kp={max_kp}'
                )
            except Exception as e:
                self.get_logger().error(f'Failed to init HLoc: {e}')
        else:
            self.get_logger().warn(
                'sfm_model_path or db_image_dir not set. '
                'Relocalization will not work.'
            )

        # ── Subscriber: 쿼리 이미지 수신 ────────────────────
        self.image_sub = self.create_subscription(
            Image,
            '/edge/relocalize/image',
            self._image_callback,
            QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE),
        )

        # ── Publisher: relocalization 결과 ───────────────────
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/sensor/relocalized_pose',
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE),
        )

        # ── Service: 수동 relocalization 트리거 ──────────────
        self.trigger_srv = self.create_service(
            Trigger, '/relocalize/trigger', self._trigger_callback
        )

        # ── Temp directory for query images ─────────────────
        self._tmp_dir = tempfile.mkdtemp(prefix='reloc_')

        self.get_logger().info('RelocalizationNode started')

    def _image_callback(self, msg: Image):
        """이미지 수신 → relocalization → pose 발행"""
        if self.localizer is None:
            return

        # 센서 ID는 header.frame_id에서 추출
        sensor_id = msg.header.frame_id or 'robot_01'

        try:
            # ROS Image → OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            # 임시 파일로 저장 (HLocLocalizer가 파일 경로를 받으므로)
            tmp_path = os.path.join(self._tmp_dir, f'{sensor_id}_query.jpg')
            cv2.imwrite(tmp_path, cv_image)

            # Relocalize
            result = self.localizer.localize(
                query_image_path=tmp_path,
                db_image_dir=self.db_image_dir,
                top_k=self.top_k,
                ransac_threshold=self.ransac_thresh,
                min_inliers=self.min_inliers,
            )

            if result is None:
                self.get_logger().warn(
                    f'Relocalization failed for {sensor_id}'
                )
                return

            # 결과를 PoseWithCovarianceStamped로 변환하여 발행
            pose_msg = self._result_to_pose_msg(result, sensor_id, msg.header.stamp)
            self.pose_pub.publish(pose_msg)

            pos = result['position_xyz']
            self.get_logger().info(
                f'Relocalized {sensor_id}: '
                f'pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), '
                f'inliers={result["num_inliers"]}, '
                f'reproj={result["mean_reproj_error_px"]:.2f}px'
            )

        except Exception as e:
            self.get_logger().error(f'Relocalization error: {e}')

    def _trigger_callback(self, request, response):
        """수동 relocalization 트리거 (디버깅용)"""
        response.success = self.localizer is not None
        response.message = (
            'Relocalization service ready'
            if self.localizer
            else 'Localizer not initialized'
        )
        return response

    def _result_to_pose_msg(self, result: dict, sensor_id: str, stamp) -> PoseWithCovarianceStamped:
        """HLoc result dict → PoseWithCovarianceStamped"""
        from scipy.spatial.transform import Rotation

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = sensor_id

        # Position
        pos = result['position_xyz']
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])

        # Orientation from T_cam_to_world
        T_c2w = np.array(result['T_cam_to_world'])
        quat = Rotation.from_matrix(T_c2w[:3, :3]).as_quat()  # [x,y,z,w]
        msg.pose.pose.orientation.x = float(quat[0])
        msg.pose.pose.orientation.y = float(quat[1])
        msg.pose.pose.orientation.z = float(quat[2])
        msg.pose.pose.orientation.w = float(quat[3])

        # Covariance: reproj error → approximate uncertainty
        reproj = result.get('mean_reproj_error_px', 5.0)
        n_inliers = result.get('num_inliers', 10)
        # Heuristic: 더 많은 inlier + 낮은 reproj = 더 높은 confidence
        sigma_t = max(0.02, reproj * 0.01)  # ~2cm minimum
        sigma_r = max(0.01, reproj * 0.005)
        cov = [0.0] * 36
        cov[0] = sigma_r ** 2   # rx
        cov[7] = sigma_r ** 2   # ry
        cov[14] = sigma_r ** 2  # rz
        cov[21] = sigma_t ** 2  # tx
        cov[28] = sigma_t ** 2  # ty
        cov[35] = sigma_t ** 2  # tz
        msg.pose.covariance = cov

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = RelocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('RelocalizationNode shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
