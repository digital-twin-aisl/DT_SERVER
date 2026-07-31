import pickle, os
import time
import queue
from typing import Any, Dict, Optional
import numpy as np
from confluent_kafka import Consumer, Producer
from confluent_kafka.error import KafkaError


class Kafka:
    def __init__(
        self,
        zone: str,
        bootstrap_servers: str,
        auto_offset_reset: str = "latest",
    ) -> None:
        
        self.bootstrap_servers = bootstrap_servers
        self.zone = zone
        self.auto_offset_reset = auto_offset_reset
        self.poll_max_records = 1000  # 더 많은 레코드를 한번에 처리
        self.sleep_on_empty_ms = 0
        self.fetch_max_wait_ms = 1  # 더 낮은 대기 시간으로 응답성 향상

        self.consumer: Optional[Consumer] = None
        self.producer: Optional[Producer] = None
        self.max_retries = 5
        self.retry_count = 0

        # 통합 큐 (모든 동기화된 데이터를 하나의 큐에)
        self.sync_queue = queue.Queue(maxsize=10)
        
        # 연속 메시지 동기화를 위한 버퍼
        self._message_sync_buffer = {}
        self._sync_timeout = 2.0  # 2초 타임아웃
        
    def connect(self) -> bool:
        while self.retry_count < self.max_retries:
            try:
                topic = f"{self.zone}_edge"

                # confluent_kafka Consumer 설정 (딕셔너리 형식)
                consumer_config = {
                    'bootstrap.servers': self.bootstrap_servers,
                    'group.id': f'rt-nav-consumer-{self.zone}',
                    'client.id': 'rt-nav-consumer',
                    
                    # 오프셋/커밋
                    'auto.offset.reset': self.auto_offset_reset,  # 실시간이면 'latest' 권장
                    'enable.auto.commit': False,                  # 처리 완료 후 수동 커밋으로 중복/유실 제어
                    
                    # 세션/하트비트
                    'session.timeout.ms': 6000,                   # 브로커 기본(10s)보다 짧게
                    'heartbeat.interval.ms': 2000,               # session_timeout_ms의 ~1/3
                    
                    # 페치(저지연)
                    'fetch.min.bytes': 1,                         # 즉시 반환
                    'fetch.wait.max.ms': 8,                       # 대기 최소화
                    'max.partition.fetch.bytes': 262144,          # 파티션당 256KB로 잘게 가져오기
                    
                    # 네트워크
                    'socket.receive.buffer.bytes': 1048576,      # 1MB 소켓 recv 버퍼
                    'socket.send.buffer.bytes': 131072,          # 128KB

                }

                self.consumer = Consumer(consumer_config)
                self.consumer.subscribe([topic])
                
                print(f"[KAFKA Zone {self.zone}] Consumer started successfully.")
                return True
            except Exception as e:
                self.retry_count += 1
                print(f"Connection attempt {self.retry_count} failed: {str(e)}")
                if self.retry_count == self.max_retries:
                    print(f"[KAFKA Zone {self.zone}] Failed to connect to Kafka broker after {self.max_retries} attempts")
                    return False
                print(f"[KAFKA Zone {self.zone}] Retrying connection... (Attempt {self.retry_count}/{self.max_retries})")
                time.sleep(2)
        return False

    def _decode_pickle(self, value: bytes) -> Optional[Dict[str, Any]]:
        try:
            return pickle.loads(value)
        except Exception:
            return None

    def _process_synchronized_messages(self, topic: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """연속으로 오는 두 메시지를 동기화하는 함수"""
        msg_time = payload.get("time")
        
        # 메시지 타입 확인 (reid vs allheatmaps)
        if 'reid' in payload:
            # 첫 번째 메시지: reid 데이터만
            reid_data = payload.get("reid")

            # 동기화 버퍼에 저장
            if msg_time not in self._message_sync_buffer:
                self._message_sync_buffer[msg_time] = {}
            self._message_sync_buffer[msg_time]['reid'] = reid_data

        if 'allheatmaps' in payload and 'roots' in payload:
            # 두 번째 메시지: allheatmaps + roots 데이터
            heatmaps_data = payload.get("allheatmaps")
            roots_data = payload.get("roots")

            # 동기화 버퍼에 저장
            if msg_time not in self._message_sync_buffer:
                self._message_sync_buffer[msg_time] = {}
            self._message_sync_buffer[msg_time]['heatmaps'] = heatmaps_data
            self._message_sync_buffer[msg_time]['roots'] = roots_data

        else:
            # 알 수 없는 메시지 타입
            return None
        
        # 현재 메시지의 timestamp에 대해서만 동기화 확인
        if msg_time in self._message_sync_buffer:
            buffer_data = self._message_sync_buffer[msg_time]
 
            # 두 메시지의 데이터가 모두 있는지 확인
            if 'reid' in buffer_data and 'heatmaps' in buffer_data and 'roots' in buffer_data:

                # 동기화된 데이터 반환
                synchronized_data = {
                    "topic": topic,
                    "time": msg_time,
                    "reid": buffer_data['reid'],
                    "roots": buffer_data['roots'],
                    "heatmaps": buffer_data['heatmaps'],
                    "is_synchronized": True
                }

                # 처리 완료된 데이터는 버퍼에서 제거
                del self._message_sync_buffer[msg_time]
                return synchronized_data
        
        # 타임아웃된 데이터 정리 (2초 이상 오래된 데이터)
        current_time = time.time()
        timeout_timestamps = []
        for timestamp, buffer_data in self._message_sync_buffer.items():
            if current_time - timestamp > self._sync_timeout:
                timeout_timestamps.append(timestamp)
        
        for timestamp in timeout_timestamps:
            del self._message_sync_buffer[timestamp]
        
        return None

    def poll_messages(self, timeout_ms: int = 100) -> bool:
        """메시지를 폴링하여 처리합니다. 메시지가 있으면 True, 없으면 False 반환"""
        if self.consumer is None:
            if not self.connect():
                print(f"[KAFKA Zone {self.zone}] Failed to connect")
                return False
        
        try:
            # confluent_kafka는 timeout을 초 단위로 받음
            timeout_sec = timeout_ms / 1000.0
            has_messages = False
            
            # 첫 번째 poll에서만 timeout 사용, 이후는 짧은 timeout으로 빠르게 처리
            for i in range(self.poll_max_records):
                # 첫 번째 poll만 전체 timeout 사용, 나머지는 0.01초로 빠르게 처리
                poll_timeout = timeout_sec if i == 0 else 0.01
                msg = self.consumer.poll(timeout=poll_timeout)
                
                if msg is None:
                    # 첫 번째 poll에서 메시지가 없으면 바로 종료
                    break
                    
                if msg.error():
                    error = msg.error()
                    # 파티션 끝에 도달 (정상적인 상황)
                    if error.code() == KafkaError._PARTITION_EOF:
                        continue
                    # 기타 에러는 로그 출력 후 계속 진행
                    else:
                        print(f"[KAFKA Zone {self.zone}] Consumer error: {error}")
                        continue
                
                has_messages = True
                topic = msg.topic()
                
                # 연속 메시지 동기화 처리
                if topic == f"{self.zone}_edge":
                    payload = self._decode_pickle(msg.value())
                    if payload:

                        synchronized_data = self._process_synchronized_messages(topic, payload)
                        if synchronized_data:
                            self._put_to_queue(self.sync_queue, synchronized_data)

            
            return has_messages
            
        except Exception as e:
            print(f"[KAFKA Zone {self.zone}] Poll error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _put_to_queue(self, target_queue: queue.Queue, event: Dict[str, Any]):
        """큐에 데이터를 넣는 헬퍼 메서드"""
        try:
            target_queue.put_nowait(event)
            return True
        except queue.Full:
            return False

    def stop(self):
        """Kafka 연결 중지"""
        pass  # 더 이상 스레드 제어가 필요 없음

    # ==================== Data Access API ====================
    def get_synchronized_data(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """동기화된 모든 데이터 가져오기 (reid, roots, heatmaps 포함)"""
        try:
            return self.sync_queue.get(timeout=timeout)
        except queue.Empty:
            return None



    def close(self) -> None:
        """연결 종료"""
        if self.consumer:
            try:
                self.consumer.close()
            except Exception:
                pass
            print(f"[KAFKA Zone {self.zone}] Consumer connection closed.")
        if self.producer:
            try:
                self.producer.flush(timeout=10)
                self.producer.close()
            except Exception:
                pass
            print(f"[KAFKA Zone {self.zone}] Producer connection closed.")

    




