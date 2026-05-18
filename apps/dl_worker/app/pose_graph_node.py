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
Pose Graph Node — GTSAM iSAM2 기반 ROS 2 중앙 Pose 최적화 노드 (Phase 4)

Subscribers:
  /robot/odom              → nav_msgs/Odometry
  /sensor/relocalized_pose → PoseWithCovarianceStamped
  /sensor/extrinsic        → TransformStamped
  /sensor/gps_anchor       → PoseWithCovarianceStamped (Phase 4)
  /sensor/uwb_anchor       → PoseWithCovarianceStamped (Phase 4)

Publishers:
  /tf                  → optimized mobile sensor poses
  /tf_static           → fixed sensor poses (from Phase 1)
  /pose_graph/stats    → std_msgs/String (Phase 4: 분산 그래프 통계 + 사이클 타이밍)

사용법:
  ros2 run dl_worker pose_graph_node --ros-args \
    -p world_frame:=world -p update_rate_hz:=10.0 \
    -p fixed_poses_json:=/path/to/sfm_all_poses.json \
    -p submap_assignments:='{"robot_01":"zone_A","cam_fixed_01":"zone_A"}' \
    -p adaptive_relinearize:=true -p target_update_ms:=50.0
"""
import json
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from scripts.phase2_pose_graph.pose_graph_manager import (
    PoseGraphManager, GTSAM_AVAILABLE,
)
from scripts.phase4_distributed.submap_manager import SubmapManager


class PoseGraphNode(Node):
    def __init__(self):
        super().__init__('pose_graph_node')

        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('update_rate_hz', 10.0)
        self.declare_parameter('fixed_poses_json', '')
        self.declare_parameter('relinearize_threshold', 0.1)
        self.declare_parameter('relinearize_skip', 10)
        # Phase 4
        self.declare_parameter('adaptive_relinearize', True)
        self.declare_parameter('target_update_ms', 50.0)
        self.declare_parameter('submap_assignments', '{}')  # JSON: {sensor_id: submap_id}
        self.declare_parameter('default_submap', 'global')
        self.declare_parameter('stats_publish_rate_hz', 1.0)

        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        update_rate = self.get_parameter('update_rate_hz').get_parameter_value().double_value
        fixed_poses_path = self.get_parameter('fixed_poses_json').get_parameter_value().string_value
        relin_thresh = self.get_parameter('relinearize_threshold').get_parameter_value().double_value
        relin_skip = self.get_parameter('relinearize_skip').get_parameter_value().integer_value
        adaptive = self.get_parameter('adaptive_relinearize').get_parameter_value().bool_value
        target_ms = self.get_parameter('target_update_ms').get_parameter_value().double_value
        submap_json = self.get_parameter('submap_assignments').get_parameter_value().string_value
        default_submap = self.get_parameter('default_submap').get_parameter_value().string_value
        stats_rate = self.get_parameter('stats_publish_rate_hz').get_parameter_value().double_value

        # Phase 4: SubmapManager가 PoseGraphManager 인스턴스(들)를 wrap
        self.pg_manager = SubmapManager(
            default_submap=default_submap,
            relinearize_threshold=relin_thresh,
            relinearize_skip=relin_skip,
            adaptive_relinearize=adaptive,
            target_update_ms=target_ms,
        )
        self._apply_submap_assignments(submap_json)

        self._odom_timesteps: dict = {}

        if fixed_poses_path:
            self._load_fixed_poses(fixed_poses_path)

        reliable_qos = QoSProfile(depth=50, reliability=QoSReliabilityPolicy.RELIABLE)

        self.odom_sub = self.create_subscription(
            Odometry, '/robot/odom', self._odom_callback, reliable_qos)
        self.reloc_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/sensor/relocalized_pose',
            self._reloc_callback, reliable_qos)
        self.extrinsic_sub = self.create_subscription(
            TransformStamped, '/sensor/extrinsic',
            self._extrinsic_callback, reliable_qos)
        # Phase 4: GPS/UWB anchor 토픽
        self.gps_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/sensor/gps_anchor',
            self._gps_callback, reliable_qos)
        self.uwb_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/sensor/uwb_anchor',
            self._uwb_callback, reliable_qos)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.stats_pub = self.create_publisher(
            String, '/pose_graph/stats',
            QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE))

        self.update_timer = self.create_timer(1.0 / update_rate, self._update_and_publish)
        if stats_rate > 0:
            self.stats_timer = self.create_timer(1.0 / stats_rate, self._publish_stats)

        self.get_logger().info(
            f"PoseGraphNode started: world={self.world_frame}, "
            f"rate={update_rate}Hz, target={target_ms}ms, "
            f"adaptive={adaptive}, GTSAM={'OK' if GTSAM_AVAILABLE else 'STUB'}")

    def _apply_submap_assignments(self, submap_json: str):
        """submap_assignments 파라미터(JSON)를 SubmapManager에 적용"""
        if not submap_json or submap_json == '{}':
            return
        try:
            mapping = json.loads(submap_json)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Invalid submap_assignments JSON: {e}')
            return
        for sensor_id, submap_id in mapping.items():
            self.pg_manager.assign_sensor(str(sensor_id), str(submap_id))
            self.get_logger().info(f'Sensor {sensor_id} → submap {submap_id}')

    def _load_fixed_poses(self, path: str):
        p = Path(path)
        if not p.exists():
            self.get_logger().warn(f'Fixed poses not found: {path}')
            return
        with open(p) as f:
            poses = json.load(f)
        for idx, (name, data) in enumerate(sorted(poses.items())):
            sid = f'sensor_fixed_{idx:02d}'
            T = np.array(data.get('T_cam_to_world', np.eye(4).tolist()))
            self.pg_manager.add_fixed_sensor_prior(sid, T, 0.001, 0.01)
        self.pg_manager.optimize_all()
        self._publish_fixed_static_tf()
        self.get_logger().info(f'Loaded {len(poses)} fixed sensor poses')

    def _publish_fixed_static_tf(self):
        transforms = []
        now = self.get_clock().now().to_msg()
        for sid, is_fixed, _, _ in self.pg_manager.iter_sensors():
            if not is_fixed:
                continue
            pose = self.pg_manager.get_current_pose(sid)
            if pose is not None:
                transforms.append(self._pose_to_tf(pose, self.world_frame, sid, now))
        if transforms:
            self.static_tf_broadcaster.sendTransform(transforms)

    def _odom_callback(self, msg: Odometry):
        sensor_id = msg.child_frame_id or 'robot_01'
        if sensor_id not in self._odom_timesteps:
            self._odom_timesteps[sensor_id] = 0
            init_pose = self._odom_to_4x4(msg)
            self.pg_manager.add_relocalization_prior(sensor_id, 0, init_pose, 0.1, 0.5)
            return
        t_from = self._odom_timesteps[sensor_id]
        t_to = t_from + 1
        self._odom_timesteps[sensor_id] = t_to
        delta = self._odom_to_4x4(msg)
        cov = msg.pose.covariance
        sr = np.sqrt(max(cov[0], 1e-6)) if sum(abs(c) for c in cov) > 0 else 0.02
        st = np.sqrt(max(cov[21], 1e-6)) if sum(abs(c) for c in cov) > 0 else 0.05
        self.pg_manager.add_odometry(sensor_id, t_from, t_to, delta, sr, st, delta)

    def _reloc_callback(self, msg: PoseWithCovarianceStamped):
        sensor_id = msg.header.frame_id
        if not sensor_id:
            return
        pose = self._pose_cov_to_4x4(msg)
        ts = self._odom_timesteps.get(sensor_id, 0)
        cov = msg.pose.covariance
        sr = np.sqrt(max(cov[0], 1e-6)) if sum(abs(c) for c in cov) > 0 else 0.05
        st = np.sqrt(max(cov[21], 1e-6)) if sum(abs(c) for c in cov) > 0 else 0.10
        self.pg_manager.add_relocalization_prior(sensor_id, ts, pose, sr, st)

    def _extrinsic_callback(self, msg: TransformStamped):
        si, sj = msg.header.frame_id, msg.child_frame_id
        if not si or not sj:
            return
        rel = self._tf_to_4x4(msg)
        ti = self._odom_timesteps.get(si, 0)
        tj = self._odom_timesteps.get(sj, 0)
        self.pg_manager.add_inter_sensor_factor(si, ti, sj, tj, rel)

    def _gps_callback(self, msg: PoseWithCovarianceStamped):
        """Phase 4: GPS anchor → add_gps_prior 라우팅"""
        sensor_id = msg.header.frame_id
        if not sensor_id:
            return
        p = msg.pose.pose.position
        position = np.array([p.x, p.y, p.z], dtype=np.float64)
        ts = self._odom_timesteps.get(sensor_id, 0)
        cov = msg.pose.covariance
        # cov는 36-tuple row-major. (3:6, 3:6)이 position covariance
        sig_pos = np.sqrt(max(cov[3 * 6 + 3], 1e-6)) if any(c != 0.0 for c in cov) else 1.0
        sig_rot = 1.0  # GPS는 orientation 없음
        self.pg_manager.add_gps_prior(sensor_id, ts, position, sig_pos, sig_rot)

    def _uwb_callback(self, msg: PoseWithCovarianceStamped):
        """Phase 4: UWB anchor → add_uwb_prior 라우팅"""
        sensor_id = msg.header.frame_id
        if not sensor_id:
            return
        p = msg.pose.pose.position
        position = np.array([p.x, p.y, p.z], dtype=np.float64)
        ts = self._odom_timesteps.get(sensor_id, 0)
        cov = msg.pose.covariance
        sig_pos = np.sqrt(max(cov[3 * 6 + 3], 1e-6)) if any(c != 0.0 for c in cov) else 0.10
        self.pg_manager.add_uwb_prior(sensor_id, ts, position, sig_pos, 1.0)

    def _update_and_publish(self):
        self.pg_manager.optimize_all()
        transforms = []
        now = self.get_clock().now().to_msg()
        for sid, is_fixed, _, _ in self.pg_manager.iter_sensors():
            if is_fixed:
                continue
            pose = self.pg_manager.get_current_pose(sid)
            if pose is not None:
                transforms.append(self._pose_to_tf(pose, self.world_frame, sid, now))
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def _publish_stats(self):
        """Phase 4: 분산 그래프 통계를 /pose_graph/stats로 발행"""
        try:
            stats = self.pg_manager.get_stats()
            msg = String()
            msg.data = json.dumps(stats, default=float)
            self.stats_pub.publish(msg)
            # 사이클 시간이 목표 초과 시 경고
            last_ms = stats.get('max_last_cycle_ms', 0.0)
            if last_ms > 0:
                self.get_logger().debug(
                    f"PoseGraph cycle: max={last_ms:.1f}ms across "
                    f"{stats['num_submaps']} submaps, "
                    f"factors={stats['total_factors']}, "
                    f"cross={stats['cross_submap_factors']}"
                )
        except Exception as e:
            self.get_logger().warn(f'stats publish failed: {e}')

    # ── Conversion utilities ────────────────────────────────
    @staticmethod
    def _quat_pos_to_4x4(qx, qy, qz, qw, tx, ty, tz):
        x, y, z, w = qx, qy, qz, qw
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def _odom_to_4x4(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        return self._quat_pos_to_4x4(q.x, q.y, q.z, q.w, p.x, p.y, p.z)

    def _pose_cov_to_4x4(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        return self._quat_pos_to_4x4(q.x, q.y, q.z, q.w, p.x, p.y, p.z)

    def _tf_to_4x4(self, msg):
        t = msg.transform.translation
        q = msg.transform.rotation
        return self._quat_pos_to_4x4(q.x, q.y, q.z, q.w, t.x, t.y, t.z)

    @staticmethod
    def _pose_to_tf(pose_4x4, parent, child, stamp):
        from scipy.spatial.transform import Rotation
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(pose_4x4[0, 3])
        t.transform.translation.y = float(pose_4x4[1, 3])
        t.transform.translation.z = float(pose_4x4[2, 3])
        quat = Rotation.from_matrix(pose_4x4[:3, :3]).as_quat()
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        return t


def main(args=None):
    rclpy.init(args=args)
    node = PoseGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('PoseGraphNode shutting down.')
    finally:
        node.get_logger().info(f'Stats: {node.pg_manager.get_stats()}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
