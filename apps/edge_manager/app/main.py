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
    async def StreamDataAndControl(self, request_iterator, context):
        device_id = None
        print("========== [gRPC] 엣지 디바이스 스트림 시작 ==========")
        try:
            # 첫 번째 메시지를 받아 device_id 식별 및 큐 등록
            first_request = await request_iterator.__anext__()
            device_id = first_request.device_id
            print(f"========== [gRPC] 엣지 연결 확인: {device_id} ==========")
            
            # 해당 엣지 전용 명령 큐(Queue) 할당
            if device_id not in edge_control_queues:
                edge_control_queues[device_id] = asyncio.Queue()

            # 첫 메시지는 데이터일 수 있으므로 그대로 처리 로직으로 넘김
            asyncio.create_task(self._process_single_request(first_request, device_id))
            
            # 송신(Server->Edge)과 수신(Edge->Server) 태스크를 분리
            # await request_iterator (단순 yield stream 사용)
            while True:
                 # queue에서 송신할 메시지 대기 중에는 request_iterator._anext() 에서도 대기하는
                 # 듀얼 루프를 통합하기 위한 병렬 wait 구조. gRPC Async 특성상 yield가 가장 직관적
                 
                 # 수신 대기 task
                 recv_task = asyncio.create_task(request_iterator.__anext__())
                 # 송신(큐) 대기 task
                 send_task = asyncio.create_task(edge_control_queues[device_id].get())

                 done, pending = await asyncio.wait(
                     [recv_task, send_task], 
                     return_when=asyncio.FIRST_COMPLETED
                 )

                 # 1. 엣지에서 올라온 메시지(Uplink) 수신 처리
                 if recv_task in done:
                     request = recv_task.result()
                     await self._process_single_request(request, device_id)
                 else:
                     recv_task.cancel() # 송신이 먼저 뜨면 수신 취소 후 다음루프 재시작

                 # 2. 서버에서 보낼 메시지 처리 (Downlink)
                 if send_task in done:
                     command_msg = send_task.result()
                     yield command_msg # gRPC Response Stream으로 엣지에 발송
                     edge_control_queues[device_id].task_done()
                 else:
                     send_task.cancel() 

        except StopAsyncIteration:
            print("========== [gRPC] 엣지가 연결을 종료함 ==========")
        except asyncio.CancelledError:
            print("========== [gRPC] 연결 종료 (Cancelled) ==========")
        except Exception as e:
            print(f"========== [gRPC] 스트리밍 오류: {e} ==========")
        finally:
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
