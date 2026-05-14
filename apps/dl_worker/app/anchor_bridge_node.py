"""
Anchor Bridge Node — GPS/UWB 절대 위치 소스 → Pose Graph 입력 변환 (Phase 4)

원시 GPS(`sensor_msgs/NavSatFix`)와 UWB(`geometry_msgs/PointStamped`) 토픽을
구독하여, world 좌표계 기준의 PoseWithCovarianceStamped로 변환 후
`/sensor/gps_anchor`, `/sensor/uwb_anchor`로 재발행합니다.

pose_graph_node가 이 토픽들을 구독하여 GPS/UWB PriorFactor를 주입합니다.

Subscribers:
  /gps/fix          → sensor_msgs/NavSatFix
  /uwb/position     → geometry_msgs/PointStamped (UWB는 표준 메시지 부재, Point 사용)

Publishers:
  /sensor/gps_anchor → geometry_msgs/PoseWithCovarianceStamped (frame_id=sensor_id)
  /sensor/uwb_anchor → geometry_msgs/PoseWithCovarianceStamped (frame_id=sensor_id)

Parameters:
  world_frame: world 좌표계 이름 (default: 'world')
  enu_origin_lat/lon/alt: ENU 원점 위경고도 (없으면 첫 GPS fix로 자동 설정)
  gps_sigma_pos: GPS 위치 표준편차 (m, default: 1.0)
  uwb_sigma_pos: UWB 위치 표준편차 (m, default: 0.10)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped

from scripts.phase4_distributed.anchor_utils import (
    geodetic_to_enu, set_enu_origin, get_enu_origin,
)


class AnchorBridgeNode(Node):
    def __init__(self):
        super().__init__('anchor_bridge_node')

        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('enu_origin_lat', float('nan'))
        self.declare_parameter('enu_origin_lon', float('nan'))
        self.declare_parameter('enu_origin_alt', 0.0)
        self.declare_parameter('gps_sigma_pos', 1.0)
        self.declare_parameter('gps_sigma_alt', 2.0)
        self.declare_parameter('uwb_sigma_pos', 0.10)
        self.declare_parameter('default_gps_sensor_id', 'robot_01')
        self.declare_parameter('default_uwb_sensor_id', 'robot_01')

        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self._gps_sigma = self.get_parameter('gps_sigma_pos').get_parameter_value().double_value
        self._gps_sigma_alt = self.get_parameter('gps_sigma_alt').get_parameter_value().double_value
        self._uwb_sigma = self.get_parameter('uwb_sigma_pos').get_parameter_value().double_value
        self._default_gps_id = self.get_parameter('default_gps_sensor_id').get_parameter_value().string_value
        self._default_uwb_id = self.get_parameter('default_uwb_sensor_id').get_parameter_value().string_value

        # 명시적 ENU 원점이 있으면 즉시 설정
        lat = self.get_parameter('enu_origin_lat').get_parameter_value().double_value
        lon = self.get_parameter('enu_origin_lon').get_parameter_value().double_value
        alt = self.get_parameter('enu_origin_alt').get_parameter_value().double_value
        if lat == lat and lon == lon:  # NaN 체크
            set_enu_origin(lat, lon, alt)
            self.get_logger().info(
                f'ENU origin from params: lat={lat:.6f}, lon={lon:.6f}, alt={alt:.2f}m'
            )

        reliable = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE)

        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self._gps_callback, reliable)
        self.uwb_sub = self.create_subscription(
            PointStamped, '/uwb/position', self._uwb_callback, reliable)

        self.gps_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/sensor/gps_anchor', reliable)
        self.uwb_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/sensor/uwb_anchor', reliable)

        self._gps_count = 0
        self._uwb_count = 0

        self.get_logger().info(
            f'AnchorBridgeNode started: world={self.world_frame}, '
            f'σ_gps={self._gps_sigma}m, σ_uwb={self._uwb_sigma}m'
        )

    def _gps_callback(self, msg: NavSatFix):
        # NavSatStatus.STATUS_NO_FIX == -1
        if msg.status.status < 0:
            return

        enu = geodetic_to_enu(msg.latitude, msg.longitude, msg.altitude)

        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id or self._default_gps_id
        out.pose.pose.position.x = float(enu[0])
        out.pose.pose.position.y = float(enu[1])
        out.pose.pose.position.z = float(enu[2])
        out.pose.pose.orientation.w = 1.0  # rotation unknown — pose_graph가 큰 sigma로 처리

        # NavSatFix.position_covariance: 3x3 row-major (m²)
        cov6 = [0.0] * 36
        nfc = msg.position_covariance
        if msg.position_covariance_type != 0 and any(c != 0.0 for c in nfc):
            # 위치 공분산 → 6x6 pose 공분산의 (3:6, 3:6) 블록에 매핑
            for i in range(3):
                for j in range(3):
                    cov6[(i + 3) * 6 + (j + 3)] = float(nfc[i * 3 + j])
            # rotation 부분은 매우 크게
            for i in range(3):
                cov6[i * 6 + i] = 1.0
        else:
            sig2 = self._gps_sigma * self._gps_sigma
            sig2_alt = self._gps_sigma_alt * self._gps_sigma_alt
            cov6[3 * 6 + 3] = sig2
            cov6[4 * 6 + 4] = sig2
            cov6[5 * 6 + 5] = sig2_alt
            for i in range(3):
                cov6[i * 6 + i] = 1.0  # rotation free

        out.pose.covariance = cov6
        self.gps_pub.publish(out)
        self._gps_count += 1
        if self._gps_count % 50 == 1:
            origin = get_enu_origin()
            self.get_logger().info(
                f'GPS#{self._gps_count}: enu=({enu[0]:.2f},{enu[1]:.2f},{enu[2]:.2f}) '
                f'origin={origin}'
            )

    def _uwb_callback(self, msg: PointStamped):
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id or self._default_uwb_id
        out.pose.pose.position.x = float(msg.point.x)
        out.pose.pose.position.y = float(msg.point.y)
        out.pose.pose.position.z = float(msg.point.z)
        out.pose.pose.orientation.w = 1.0

        sig2 = self._uwb_sigma * self._uwb_sigma
        cov6 = [0.0] * 36
        cov6[3 * 6 + 3] = sig2
        cov6[4 * 6 + 4] = sig2
        cov6[5 * 6 + 5] = sig2
        for i in range(3):
            cov6[i * 6 + i] = 1.0
        out.pose.covariance = cov6

        self.uwb_pub.publish(out)
        self._uwb_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = AnchorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('AnchorBridgeNode shutting down.')
    finally:
        node.get_logger().info(
            f'Final: GPS={node._gps_count}, UWB={node._uwb_count}'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
