"""
VIO Bridge Node — Visual-Inertial Odometry ROS 2 인터페이스

ORB-SLAM3, Kimera, VINS-Fusion 등 VIO 시스템의 출력을
ROS 2 nav_msgs/Odometry로 변환하여 pose_graph_node에 제공합니다.

지원 모드:
  1. 'orbslam3': ORB-SLAM3 ROS 2 wrapper 출력 구독
  2. 'kimera':   Kimera-VIO ROS 2 출력 구독
  3. 'stub':     테스트용 시뮬레이션 odometry 생성

Subscriber (입력):
  /camera/image_raw   → sensor_msgs/Image (VIO 입력)
  /imu/data           → sensor_msgs/Imu   (VIO IMU 입력)
  
  또는 VIO 시스템이 이미 ROS 2 노드인 경우:
  /vio/pose           → geometry_msgs/PoseStamped (VIO 출력)

Publisher (출력):
  /robot/odom          → nav_msgs/Odometry (pose_graph_node 입력)

사용법:
  ros2 run dl_worker vio_bridge_node --ros-args \
    -p vio_mode:=orbslam3 \
    -p sensor_frame:=robot_01/base_link
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, Imu
import math


class VIOBridgeNode(Node):
    """
    Visual-Inertial Odometry Bridge Node
    
    VIO 백엔드 시스템의 출력을 표준 nav_msgs/Odometry로 변환하여
    pose_graph_node에 연결합니다.
    """

    SUPPORTED_MODES = ('orbslam3', 'kimera', 'vins', 'stub')

    def __init__(self):
        super().__init__('vio_bridge_node')

        # ── Parameters ──────────────────────────────────────
        self.declare_parameter('vio_mode', 'stub')
        self.declare_parameter('sensor_frame', 'robot_01/base_link')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('odom_topic', '/robot/odom')
        # Stub mode parameters
        self.declare_parameter('stub_velocity', 0.1)       # m/s
        self.declare_parameter('stub_angular_vel', 0.05)   # rad/s
        self.declare_parameter('stub_rate_hz', 10.0)

        self.vio_mode = self.get_parameter('vio_mode').get_parameter_value().string_value
        self.sensor_frame = self.get_parameter('sensor_frame').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.stub_vel = self.get_parameter('stub_velocity').get_parameter_value().double_value
        self.stub_ang = self.get_parameter('stub_angular_vel').get_parameter_value().double_value
        stub_rate = self.get_parameter('stub_rate_hz').get_parameter_value().double_value

        if self.vio_mode not in self.SUPPORTED_MODES:
            self.get_logger().error(
                f'Unknown vio_mode: {self.vio_mode}. '
                f'Supported: {self.SUPPORTED_MODES}'
            )

        # ── Publisher ───────────────────────────────────────
        self.odom_pub = self.create_publisher(
            Odometry, odom_topic,
            QoSProfile(depth=50, reliability=QoSReliabilityPolicy.RELIABLE))

        # ── Subscribers (mode-dependent) ────────────────────
        reliable_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

        if self.vio_mode == 'orbslam3':
            # ORB-SLAM3 publishes on /orb_slam3/camera_pose
            self.pose_sub = self.create_subscription(
                PoseStamped, '/orb_slam3/camera_pose',
                self._vio_pose_callback, reliable_qos)
            self.get_logger().info('VIO mode: ORB-SLAM3 (subscribing to /orb_slam3/camera_pose)')

        elif self.vio_mode == 'kimera':
            # Kimera-VIO publishes odometry directly
            self.pose_sub = self.create_subscription(
                Odometry, '/kimera_vio_ros/odometry',
                self._vio_odom_passthrough, reliable_qos)
            self.get_logger().info('VIO mode: Kimera (subscribing to /kimera_vio_ros/odometry)')

        elif self.vio_mode == 'vins':
            # VINS-Fusion publishes odometry
            self.pose_sub = self.create_subscription(
                Odometry, '/vins_estimator/odometry',
                self._vio_odom_passthrough, reliable_qos)
            self.get_logger().info('VIO mode: VINS-Fusion (subscribing to /vins_estimator/odometry)')

        elif self.vio_mode == 'stub':
            # Simulated circular motion for testing
            self._stub_time = 0.0
            self._stub_dt = 1.0 / stub_rate
            self._prev_pose = np.eye(4)
            self.stub_timer = self.create_timer(self._stub_dt, self._stub_generate)
            self.get_logger().info(
                f'VIO mode: STUB (simulated odometry at {stub_rate}Hz)')

        self.get_logger().info(
            f'VIOBridgeNode started: mode={self.vio_mode}, '
            f'frame={self.sensor_frame}, odom_topic={odom_topic}')

    # ─────────────────────────────────────────────────────────
    # VIO Backend Callbacks
    # ─────────────────────────────────────────────────────────

    def _vio_pose_callback(self, msg: PoseStamped):
        """
        VIO PoseStamped → Odometry 변환 (ORB-SLAM3 등)
        """
        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.sensor_frame
        odom.pose.pose = msg.pose

        # Default covariance (VIO 시스템이 제공하지 않는 경우)
        cov = [0.0] * 36
        cov[0] = 0.01 ** 2   # orientation ~0.6°
        cov[7] = 0.01 ** 2
        cov[14] = 0.01 ** 2
        cov[21] = 0.03 ** 2  # position ~3cm
        cov[28] = 0.03 ** 2
        cov[35] = 0.03 ** 2
        odom.pose.covariance = cov

        self.odom_pub.publish(odom)

    def _vio_odom_passthrough(self, msg: Odometry):
        """
        VIO Odometry 패스스루 (Kimera, VINS-Fusion)
        frame_id만 재설정하여 전달합니다.
        """
        msg.header.frame_id = self.world_frame
        msg.child_frame_id = self.sensor_frame
        self.odom_pub.publish(msg)

    # ─────────────────────────────────────────────────────────
    # Stub Mode: 시뮬레이션 Odometry 생성
    # ─────────────────────────────────────────────────────────

    def _stub_generate(self):
        """
        테스트용 시뮬레이션 odometry 생성
        
        반지름 2m의 원형 경로를 따라 이동하는 로봇을 시뮬레이션합니다.
        """
        self._stub_time += self._stub_dt

        # 원형 궤적 (반지름 2m)
        radius = 2.0
        omega = self.stub_ang  # angular velocity
        t = self._stub_time

        x = radius * math.cos(omega * t)
        y = radius * math.sin(omega * t)
        z = 0.5  # fixed height
        yaw = omega * t + math.pi / 2  # tangent direction

        # Yaw → quaternion
        qw = math.cos(yaw / 2)
        qz = math.sin(yaw / 2)

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.sensor_frame

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Simulated covariance (increases with time → drift simulation)
        drift_factor = min(self._stub_time * 0.001, 0.1)  # max 10cm
        cov = [0.0] * 36
        cov[0] = (0.01 + drift_factor) ** 2
        cov[7] = (0.01 + drift_factor) ** 2
        cov[14] = (0.01 + drift_factor) ** 2
        cov[21] = (0.02 + drift_factor) ** 2
        cov[28] = (0.02 + drift_factor) ** 2
        cov[35] = (0.02 + drift_factor) ** 2
        odom.pose.covariance = cov

        self.odom_pub.publish(odom)

        # Log every 50 frames
        if int(self._stub_time / self._stub_dt) % 50 == 0:
            self.get_logger().info(
                f'Stub odom: t={t:.1f}s pos=({x:.3f}, {y:.3f}, {z:.3f})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = VIOBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('VIOBridgeNode shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
