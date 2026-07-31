import pickle
import time
import uuid
import os
from confluent_kafka import Producer, Consumer, KafkaException
from multiprocessing import Process

class KafkaConsumer(Process):
    def __init__(self, bootstrap_servers=None, input_flag=None, auto_offset_reset='latest', group_id=None):
        super().__init__(daemon=False)
        self.bootstrap_servers = bootstrap_servers
        self.input_flag = input_flag
        self.auto_offset_reset = auto_offset_reset
        # 1:N 통신을 위해 각 Consumer마다 고유한 group_id 생성
        self.group_id = group_id if group_id else f'rt-nav-consumer-{os.getpid()}-{uuid.uuid4().hex[:8]}'
        self.consumer = None
        self.max_retries = 5
        self.retry_count = 0
        self.chunk_size = 1000
        self.topic = 'flag'
    def run(self):
        self.retry_count = 0
        while self.retry_count < self.max_retries:
            try:
                consumer_config = {
                    'bootstrap.servers': self.bootstrap_servers,
                    'group.id': self.group_id,  # 각 Consumer마다 고유한 group_id 사용 (1:N 통신)
                    'client.id': f'rt-nav-consumer-{os.getpid()}',
                    
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
                
                # 토픽이 생성될 때까지 대기 (auto.create.topics.enable=true인 경우)
                print(f"[KAFKA Consumer] Subscribing to topic: {self.topic} (group_id: {self.group_id})")
                self.consumer.subscribe([self.topic])
                
                # 토픽이 준비될 때까지 짧은 대기
                time.sleep(1)
                
                # 연결 성공 후 메시지 소비 시작
                print(f"[KAFKA Consumer] Consumer started successfully (group_id: {self.group_id})")
                self.consume()
                return True
            except Exception as e:
                self.retry_count += 1
                print(f"Connection attempt {self.retry_count} failed: {str(e)}")
                if self.retry_count == self.max_retries:
                    print(f"[KAFKA Consumer] Failed to connect to Kafka broker after {self.max_retries} attempts")
                    return False
                print(f"[KAFKA Consumer] Retrying connection... (Attempt {self.retry_count}/{self.max_retries})")
                time.sleep(2)
        return False

    def consume(self):
        '''
        수신 
        True or False boolean value
        '''
        while True:
            try:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    error = msg.error()
                    # 토픽이 아직 생성되지 않은 경우 재시도 (auto.create.topics.enable=true인 경우)
                    error_str = str(error)
                    if "UNKNOWN_TOPIC_OR_PART" in error_str or "Unknown topic or partition" in error_str:
                        print(f"[KAFKA Consumer] Topic '{self.topic}' not available yet, waiting...")
                        time.sleep(2)
                        continue
                    print(f"[KAFKA Consumer] Error: {error}")
                    continue
                topic = msg.topic()
                if topic == self.topic:
                    payload = pickle.loads(msg.value())
                    self.input_flag.value = payload

                    # 메시지 처리 완료 후 수동 커밋
                    self.consumer.commit(msg)
                
            except Exception as e:
                print(f"[KAFKA Consumer] Error consuming message: {e}")
                time.sleep(2)

class KafkaProducer:
    def __init__(self, zone, bootstrap_servers=None):
        self.bootstrap_servers = bootstrap_servers
        self.zone = zone

        self.producer = None
        self.max_retries = 5
        self.retry_count = 0
        self.chunk_size = 1000

    def _delivery_report(self, err, msg):
        if err is not None:
            print(f"Delivery failed: {err}")

    def connect(self):
        conf = {
            "bootstrap.servers": self.bootstrap_servers,
            "acks": "1",
            "enable.idempotence": False,
            "compression.type": "lz4",
            "linger.ms": 0,
            "batch.num.messages": 10000,
            "retries": 1,
            "request.timeout.ms": 30000,
            "retry.backoff.ms": 100,
            "max.in.flight.requests.per.connection": 5,
            "message.max.bytes": 104857600,
            "socket.keepalive.enable": True,
        }

        while self.retry_count < self.max_retries:
            try:
                self.producer = Producer(conf)
                self.producer.list_topics(timeout=10)
                print("Kafka Producer started successfully.")
                return True
            except KafkaException as e:
                self.retry_count += 1
                print(f"Connection attempt {self.retry_count} failed: {str(e)}")
                if self.retry_count == self.max_retries:
                    print(f"Failed to connect to Kafka broker after {self.max_retries} attempts")
                    return False
                print(f"Retrying connection... (Attempt {self.retry_count}/{self.max_retries})")
                time.sleep(2)
            except Exception as e:
                self.retry_count += 1
                print(f"Unexpected error on attempt {self.retry_count}: {str(e)}")
                if self.retry_count == self.max_retries:
                    print(f"Failed to connect to Kafka broker after {self.max_retries} attempts")
                    return False
                print(f"Retrying connection... (Attempt {self.retry_count}/{self.max_retries})")
                time.sleep(2)
        return False

    def send(self, topic, reid, root, allheatmap, timestamp):
        '''
        데이터 전송 api
        {
            "time": timestamp, # float64
            "reid": reid, #
        },
        {
            "time": timestamp, # float64
            "allheatmaps": allheatmap, # numpy array(1,15,128,240) uint8
            "roots": root # numpy array(1,10,5) float32
        }
        '''
        if self.producer is None:
            return False
        
        try:
            payloads = [
                {"time": timestamp, "reid": reid},
                {"time": timestamp, "allheatmaps": allheatmap, "roots": root},
            ]
            for p in payloads:
                self.producer.produce(
                    topic=topic,
                    value=pickle.dumps(p),
                    on_delivery=self._delivery_report,
                )
            self.producer.flush(timeout=30.0)
            return True
        except BufferError:
            try:
                self.producer.flush(timeout=30.0)
                for p in payloads:
                    self.producer.produce(topic=topic, value=pickle.dumps(p), on_delivery=self._delivery_report)
                self.producer.flush(timeout=30.0)
                return True
            except Exception:
                return False
        except Exception:
            return False
