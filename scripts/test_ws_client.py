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

import asyncio
import websockets

async def listen_sim_backend():
    uri = "ws://sim_backend:80/ws/sim"
    print(f"🌍 접속 시도: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ [Client] Sim Backend WebSocket 서버와 연결되었습니다. 3D 메타데이터를 대기합니다...")
            print("=====================================================")
            
            while True:
                # 서버에서 브로드캐스트하는 추론 결과를 계속 리스닝
                message = await websocket.recv()
                print(f"\n📥 [수신 데이터]:\n{message}")
                print("=====================================================")
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ [Client] 서버 연결이 종료되었습니다: {e}")
    except Exception as e:
        print(f"\n❌ [Client] 에러 발생: {e}")

if __name__ == "__main__":
    try:
         asyncio.run(listen_sim_backend())
    except KeyboardInterrupt:
         print("종료됨")