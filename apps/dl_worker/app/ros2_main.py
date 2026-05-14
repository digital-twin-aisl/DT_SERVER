"""
DL Inference ROS 2 Node — TF2 기반 실시간 센서 Pose 조회

[Phase 1 리팩토링]
- 기존: HTTP 호출로 camera_manager에서 정적 캘리브레이션 1회 캐싱
- 개선: tf2_ros.TransformListener를 통해 /tf, /tf_static에서
        센서 pose를 실시간 lookup → 이동형 센서, 온라인 캘리브레이션 지원
"""
import os
import asyncio

import numpy as np
import rclpy
from rclpy.node import Node
from dt_interfaces.msg import TensorMsg
from geometry_msgs.msg import TransformStamped, PoseArray, Pose
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.qos import qos_profile_sensor_data

from app.core.inference import DLInferencer


class DLInferenceNode(Node):
    def __init__(self):
        super().__init__('dl_inference_node')

        # ── 1. TF2 Buffer & Listener ────────────────────────
        # /tf_static (고정 센서 — StaticPosePublisher가 발행) 및
        # /tf (이동 센서 — pose_graph_node가 발행)를 모두 수신
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── 2. Subscriber: 엣지 텐서 수신 ────────────────────
        # [Architecture V2] 기존 ByteMultiArray + Protobuf 대신 
        # 커스텀 TensorMsg를 직접 구독하여 타입 안전성 및 인트로스펙션 확보
        # Best-Effort QoS 사용 (ARCHITECTURE.md 참조)
        self.subscription = self.create_subscription(
            TensorMsg,
            '/edge/camera/tensor',
            self.tensor_callback,
            qos_profile_sensor_data
        )

        # ── 3. Publisher: 추론된 3D Pose → /tf 및 PoseArray ─────────────
        self.tf_publisher = self.create_publisher(TFMessage, '/tf', 10)
        self.pose_array_publisher = self.create_publisher(PoseArray, '/person_poses', 10)

        # ── 4. Inference module ─────────────────────────────
        self.inferencer = DLInferencer()

        # ── 5. world frame 설정 ─────────────────────────────
        self.declare_parameter('world_frame', 'world')
        self.world_frame = (
            self.get_parameter('world_frame')
            .get_parameter_value().string_value
        )

        self.get_logger().info(
            f"DL Inference Node started. "
            f"Using TF2 for sensor pose lookup (world_frame={self.world_frame})"
        )

    # ────────────────────────────────────────────────────────
    # TF2 기반 센서 Pose 조회 (HTTP 호출 제거)
    # ────────────────────────────────────────────────────────

    def get_sensor_pose(self, sensor_frame: str) -> TransformStamped:
        """
        TF2 Buffer에서 센서의 전역 pose를 실시간 조회

        Args:
            sensor_frame: TF child_frame_id (e.g. 'sensor_fixed_01', 'robot_01/camera_front')

        Returns:
            TransformStamped: world → sensor_frame 변환

        Raises:
            TransformException: TF를 아직 받지 못했거나, 해당 frame이 없는 경우
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                sensor_frame,
                rclpy.time.Time()  # latest available
            )
            return transform
        except TransformException as e:
            self.get_logger().warn(
                f'TF lookup failed ({self.world_frame} → {sensor_frame}): {e}'
            )
            raise

    def transform_to_extrinsic_matrix(self, tf: TransformStamped) -> np.ndarray:
        """
        TransformStamped를 4x4 extrinsic matrix로 변환

        Returns:
            4x4 numpy array (world-to-camera transform)
        """
        t = tf.transform.translation
        q = tf.transform.rotation

        # Quaternion (x, y, z, w) → rotation matrix
        R = self._quat_to_rotmat(q.x, q.y, q.z, q.w)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    @staticmethod
    def _quat_to_rotmat(x, y, z, w) -> np.ndarray:
        """Quaternion (x, y, z, w) → 3x3 rotation matrix"""
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
        ], dtype=np.float64)

    # ────────────────────────────────────────────────────────
    # 추론 파이프라인
    # ────────────────────────────────────────────────────────

    def process_message(self, msg: TensorMsg):
        """텐서 데이터 수신 → 추론 → TF 발행 (동기)"""
        # [Architecture V2] Protobuf 파싱 과정 제거 (ROS 2 Native Message 직접 사용)
        
        # 센서 frame ID 결정 (메시지 메타데이터에서 추출)
        sensor_frame = f'sensor_fixed_{msg.camera_id:02d}'

        # 1. TF2에서 센서 pose 실시간 조회
        try:
            sensor_tf = self.get_sensor_pose(sensor_frame)
            T_world_cam = self.transform_to_extrinsic_matrix(sensor_tf)
        except TransformException:
            self.get_logger().warn(
                f'센서 {sensor_frame}의 TF를 찾을 수 없습니다. '
                f'StaticPosePublisher가 실행 중인지 확인하세요.'
            )
            # Fallback: 단위 행렬 (원점에 위치)
            T_world_cam = np.eye(4, dtype=np.float64)

        # 2. 추론 수행 (T_world_cam을 직접 전달)
        inference_result = self.inferencer.process_tensor(
            feature_map=msg.feature_map,
            voxel_data=msg.voxel_data,
            camera_id=msg.camera_id,
            calib_params=self._matrix_to_calib_dict(T_world_cam)
        )

        # 3. 추론 결과를 /tf 및 PoseArray로 발행
        tf_msg = TFMessage()
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = self.world_frame

        for person in inference_result.get("persons", []):
            # TransformStamped 구성
            t = TransformStamped()
            t.header.stamp = pose_array_msg.header.stamp
            t.header.frame_id = self.world_frame
            t.child_frame_id = f"person_{person.get('track_id', 0)}"

            pos = person.get("position", {})
            t.transform.translation.x = float(pos.get("x", 0.0))
            t.transform.translation.y = float(pos.get("y", 0.0))
            t.transform.translation.z = float(pos.get("z", 0.0))

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            tf_msg.transforms.append(t)

            # Pose 구성
            pose = Pose()
            pose.position.x = t.transform.translation.x
            pose.position.y = t.transform.translation.y
            pose.position.z = t.transform.translation.z
            pose.orientation = t.transform.rotation
            pose_array_msg.poses.append(pose)

        if tf_msg.transforms:
            self.tf_publisher.publish(tf_msg)
            self.pose_array_publisher.publish(pose_array_msg)
            self.get_logger().info(
                f"Published TF and PoseArray for {len(tf_msg.transforms)} persons "
                f"from {sensor_frame} (Frame: {msg.frame_id})"
            )

    @staticmethod
    def _matrix_to_calib_dict(T: np.ndarray) -> dict:
        """
        4x4 transform → legacy calib_params dict (호환용)

        기존 inference.py의 transform_to_world()가 RPY+XYZ dict를
        받으므로, 4x4 행렬에서 변환합니다.

        Phase 2: T_world_cam_4x4 키를 함께 전달하여
        inference.py가 4x4 행렬을 직접 사용할 수 있도록 지원.
        """
        from scipy.spatial.transform import Rotation
        R = T[:3, :3]
        r = Rotation.from_matrix(R)
        roll, pitch, yaw = r.as_euler('xyz')
        return {
            'x': float(T[0, 3]),
            'y': float(T[1, 3]),
            'z': float(T[2, 3]),
            'roll': float(roll),
            'pitch': float(pitch),
            'yaw': float(yaw),
            # Phase 2: 4x4 행렬 직접 전달 (inference.py에서 우선 사용)
            'T_world_cam_4x4': T.tolist(),
        }

    def tensor_callback(self, msg: TensorMsg):
        """ROS 2 커스텀 메시지 수신 콜백"""
        self.process_message(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DLInferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by Keyboard Interrupt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
