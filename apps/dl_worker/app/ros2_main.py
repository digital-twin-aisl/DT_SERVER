import os
import json
import asyncio
import httpx

import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

from app.protos import edge_communication_pb2
from app.core.inference import DLInferencer

class DLInferenceNode(Node):
    def __init__(self):
        super().__init__('dl_inference_node')
        
        # 1. 수신부 (Subscriber): 엣지에서 들어오는 텐서 데이터(직렬화된 Protobuf 바이트) 수신
        # (임시로 ByteMultiArray를 통해 바이트 수신. 추후 커스텀 Msg로 변경 가능)
        self.subscription = self.create_subscription(
            ByteMultiArray,
            '/edge/camera/tensor',
            self.tensor_callback,
            10 # QoS (기본적으로 Reliable 맵핑, 커스텀 프로파일로 Best Effort 지정 가능)
        )
        
        # 2. 송신부 (Publisher): 추론이 완료된 3D Pose를 ROS 2 표준인 TFMessage로 발행
        self.tf_publisher = self.create_publisher(TFMessage, '/tf', 10)
        
        # 3. 싱글톤 추론 모듈 로드
        self.inferencer = DLInferencer()
        self.camera_manager_url = os.getenv("CAMERA_MANAGER_URL", "http://camera_manager:80")
        
        # 4. Redis를 걷어내고 메모리 내 캘리브레이션 캐싱 사용
        self.calib_cache = {}
        
        self.get_logger().info("DL Inference ROS 2 Node has been started successfully.")

    async def get_camera_calibration(self, edge_id: str, camera_id: int) -> dict:
        """Camera Manager에서 캘리브레이션 정보를 동기화 (비동기)"""
        cache_key = f"{edge_id}_{camera_id}"
        if cache_key in self.calib_cache:
            return self.calib_cache[cache_key]
            
        self.get_logger().info(f"[{cache_key}] Fetching calibration from camera_manager...")
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.camera_manager_url}/api/cameras/{edge_id}/{camera_id}/calibration",
                    timeout=5.0
                )
                resp.raise_for_status()
                calibration_data = resp.json()
        except Exception as e:
            self.get_logger().warn(f"camera_manager call failed: {e}. Using default calibration.")
            calibration_data = {
                "x": 0.0, "y": 0.0, "z": 0.0,
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0
            }
        
        self.calib_cache[cache_key] = calibration_data
        return calibration_data

    async def process_message_async(self, payload_bytes):
        # Protobuf 파싱
        video_data = edge_communication_pb2.VideoFeatureData()
        video_data.ParseFromString(payload_bytes)
        
        device_id = "edge_device_default" # 실제로는 메타데이터에서 추출
        
        # 1. 캘리브레이션 파라미터 획득
        calib_params = await self.get_camera_calibration(device_id, video_data.camera_id)
        
        # 2. 딥러닝 추론 및 좌표 변환
        inference_result = self.inferencer.process_tensor(
            feature_map=video_data.feature_map, 
            voxel_data=video_data.voxel_data,
            camera_id=video_data.camera_id,
            calib_params=calib_params
        )
        
        # 3. 추론된 결과를 ROS 2 표준 TFMessage로 변환하여 Publish (Isaac Sim 직결용)
        tf_msg = TFMessage()
        
        for person in inference_result.get("persons", []):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "world"  # 기준 좌표계 (Absolute World)
            t.child_frame_id = f"person_{person.get('person_id', 0)}" # 대상 아바타 ID
            
            # 추론된 3D 위치(Translation) 삽입
            t.transform.translation.x = float(person.get("pose", {}).get("x", 0.0))
            t.transform.translation.y = float(person.get("pose", {}).get("y", 0.0))
            t.transform.translation.z = float(person.get("pose", {}).get("z", 0.0))
            
            # 임시 기본 회전(Rotation) 삽입 (Quaternion)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            
            tf_msg.transforms.append(t)
            
        if tf_msg.transforms:
            self.tf_publisher.publish(tf_msg)
            self.get_logger().info(f"Published TF for {len(tf_msg.transforms)} persons. (Frame: {video_data.frame_id})")

    def tensor_callback(self, msg):
        """ROS 2 토픽 수신 콜백 (비동기 루프 호출)"""
        # ROS 2 rclpy는 기본적으로 동기 콜백이므로, asyncio 루프를 활용해 비동기 HTTP 통신 등을 처리합니다.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        payload_bytes = bytes(msg.data)
        loop.run_until_complete(self.process_message_async(payload_bytes))

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
