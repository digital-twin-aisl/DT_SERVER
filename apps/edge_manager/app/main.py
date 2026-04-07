import asyncio
import os
import json
from fastapi import FastAPI
import grpc
from grpc.aio import server as grpc_server
import redis.asyncio as redis

from app.protos import edge_communication_pb2
from app.protos import edge_communication_pb2_grpc

# Redis 연결 설정 (환경 변수 우선, 기본값 redis://redis:6379)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=False)
redis_pubsub = redis_client.pubsub()

# 엣지별로 전송할 명령 버퍼를 담아두기 위한 전역 Dict (key: device_id, value: asyncio.Queue)
edge_control_queues = {}

class EdgeManagerServicer(edge_communication_pb2_grpc.EdgeManagerServicer):
    
    async def _safe_recv(self, request_iterator):
        """gRPC iterator의 내부 StopAsyncIteration 예외를 안전하게 캐치합니다."""
        try:
            return await request_iterator.__anext__()
        except StopAsyncIteration:
            return grpc.aio.EOF

    async def StreamDataAndControl(self, request_iterator, context):
        device_id = None
        print("========== [gRPC] 엣지 디바이스 스트림 시작 ==========")
        
        # 1. 안전한 비동기 태스크 생성 (__anext__() 호출 시 발생하는 예외 방지)
        receive_task = asyncio.create_task(self._safe_recv(request_iterator))
        send_task = None
        
        try:
            while True:
                # 엣지 큐가 할당되었으나 send_task가 없으면 대기 태스크 생성
                if send_task is None and device_id is not None:
                    send_task = asyncio.create_task(edge_control_queues[device_id].get())
                
                tasks = [receive_task]
                if send_task:
                    tasks.append(send_task)
                    
                done, pending = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # --- 수신(Uplink) 처리 ---
                if receive_task in done:
                    request = receive_task.result()
                    
                    # 연결 강제 종료 또는 클라이언트가 스트림을 닫았을 때
                    if request is grpc.aio.EOF or request is None:
                        print(f"========== [gRPC] 스트림 EOF 도달 (정상 종료) ==========")
                        break
                    
                    # 첫 연결 시 device_id 저장
                    if not device_id and request.device_id:
                        device_id = request.device_id
                        print(f"========== [gRPC] 엣지 연결 확인: {device_id} ==========")
                        # 해당 엣지 전용 명령 큐 할당
                        if device_id not in edge_control_queues:
                            edge_control_queues[device_id] = asyncio.Queue()

                    # 개별 데이터 처리 분기 (Redis 발행 등)
                    await self._process_single_request(request, device_id)
                    
                    # 다음 데이터 수신을 위해 루프 갱신
                    receive_task = asyncio.create_task(self._safe_recv(request_iterator))

                # --- 송신(Downlink) 처리 ---
                if send_task and send_task in done:
                    command_msg = send_task.result()
                    await context.write(command_msg)
                    
                    # 다음 명령을 기다리기 위해 태스크 갱신
                    send_task = asyncio.create_task(edge_control_queues[device_id].get())

        except asyncio.CancelledError:
            print("========== [gRPC] 연결 종료 (Cancelled) ==========")
        except Exception as e:
            print(f"========== [gRPC] 스트리밍 오류: {e} ==========")
        finally:
            # 잔여 태스크 취소
            if not receive_task.done():
                receive_task.cancel()
            if send_task and not send_task.done():
                send_task.cancel()
                
            # 스트림 종료 시 큐 정리
            if device_id and device_id in edge_control_queues:
                del edge_control_queues[device_id]
                print(f"========== [gRPC] 엣지 연결 해제 완료: {device_id} ==========")

    async def _process_single_request(self, request, device_id):
        """개별 요청(데이터 업링크)에 대한 처리 (Video/ONVIF 등)"""
        try:
            if request.HasField("video_data"):
                # Redis Stream으로 분석 컨슈머에게 전달
                await redis_client.xadd(
                    name="stream:video_data",
                    fields={
                        b"device_id": device_id.encode('utf-8'),
                        b"payload": request.video_data.SerializeToString()
                    },
                    maxlen=100, 
                    approximate=True
                )
            elif request.HasField("onvif_data"):
                # ONVIF 데이터는 Pub/Sub 브로드캐스트
                await redis_client.publish("pubsub:onvif_metadata", request.onvif_data.SerializeToString())
            elif request.HasField("heartbeat"):
                pass
        except Exception as e:
            print(f"[{device_id}] 개별 요청 처리 중 오류: {e}")

# --- 여기까지 EdgeManagerServicer 클래스 정의 종료 ---

# --- Redis Subscriber (백그라운드 워커) ---
async def redis_control_subscriber():
    """
    백그라운드에서 실행되며 Redis Pub/Sub 채널 'control:edge:*'의 명령을 구독하고,
    해당하는 엣지의 asyncio.Queue()로 명령을 밀어넣습니다.
    """
    print("========== [Redis] 제어 명령 리스너(Subscriber) 시작 ==========")
    await redis_pubsub.psubscribe("control:edge:*")
    
    try:
        async for message in redis_pubsub.listen():
            if message["type"] == "pmessage":
                channel = message["channel"].decode('utf-8')
                data = message["data"].decode('utf-8')
                
                # 채널명 파싱: "control:edge:{device_id}" -> 마지막 파트 추출
                target_device_id = channel.split(":")[-1]
                
                if target_device_id in edge_control_queues:
                    # JSON 분해 및 gRPC 응답 형식으로 컨버팅
                    parsed_data = json.loads(data)
                    grpc_command = edge_communication_pb2.ServerControlResponse(
                        command=parsed_data.get("command", "UNKNOWN"),
                        parameters=json.dumps(parsed_data) # 전체 데이터를 JSON 파라미터로
                    )
                    
                    # 엣지 큐에 넣어서 _send_loop 가 꺼내가도록 함 (Downlink 발송)
                    await edge_control_queues[target_device_id].put(grpc_command)
                    print(f"[Redis->gRPC] 엣지 {target_device_id} 로 명령 전송 큐에 추가: {parsed_data.get('command')}")
                else:
                    print(f"[Redis] 엣지({target_device_id})가 연결되어 있지 않아 명령 무시됨.")
    except Exception as e:
        print(f"========== [Redis Sub] 에러 발생: {e} ==========")

async def serve_grpc():

    server = grpc_server()
    edge_communication_pb2_grpc.add_EdgeManagerServicer_to_server(EdgeManagerServicer(), server)
    server.add_insecure_port('[::]:50051')
    
    await server.start()
    print("gRPC 서버가 포트 50051에서 시작되었습니다.")
    await server.wait_for_termination()

app = FastAPI(title="Edge Manager Service")

@app.on_event("startup")
async def startup_event():
    # FastAPI 시작 시 백그라운드 태스크 두 개 생성
    
    # 1. gRPC 수신/통신 서버 시작
    asyncio.create_task(serve_grpc())
    
    # 2. Redis로부터 받아온 제어 명령을 gRPC로 푸시해주는 리스너 시작
    asyncio.create_task(redis_control_subscriber())

@app.get("/")
def read_root():
    return {"message": "Hello from Edge Manager. HTTP and gRPC are running."}
