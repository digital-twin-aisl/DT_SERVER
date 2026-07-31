import cv2, time,threading
import os.path as osp
import glob
from multiprocessing import Process, Event, Pipe
from multiprocessing import shared_memory
import numpy as np

class ExampleDataset(Process):
    def __init__(self, paths, stop_event, conn, input_flag):
        super().__init__(daemon=False)
        self.paths = sorted(glob.glob(osp.join(paths, 'hdVideos', '*.mp4')))
        self.stop_event = stop_event
        self.conn = conn
        self.flag=input_flag
        self.n = len(self.paths)
        self.shms   = [None] * self.n
        self.shapes = [None] * self.n
        self.dtypes = [None] * self.n
    def run(self):


        caps = [cv2.VideoCapture(p) for p in self.paths]
        try:
            # 초기 프레임로 해상도 파악 및 공유메모리 생성
            for i, cap in enumerate(caps):
                ok, frame = cap.read()
                if not ok:
                    self.conn.send(("ERR", i))
                    continue
                h, w, c = frame.shape
                shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                self.shms[i]   = shm
                self.shapes[i] = (h, w, c)
                self.dtypes[i] = frame.dtype

                # 메타 전송
                self.conn.send(("OK", i, shm.name, (h, w, c), frame.dtype.str))

                # 초기 프레임 기록
                buf = np.ndarray((h, w, c), dtype=frame.dtype, buffer=shm.buf)
                buf[:] = frame

            # 단일 루프: 모든 소스를 라운드로빈으로 읽고 각 shm에 덮어쓰기
            while not self.stop_event.is_set():
                any_alive = False

                if not self.flag.value:
                    time.sleep(0.01)
                    continue

                for i, cap in enumerate(caps):
                    if self.shms[i] is None:
                        continue
                    ret, frame = cap.read()
                
                    if not ret:
                        continue
                    any_alive = True
                    h, w, c = self.shapes[i]
                    dtype = self.dtypes[i]
                    if frame.shape != (h, w, c) or frame.dtype != dtype:
                        frame = cv2.resize(frame, (w, h))
                        if frame.dtype != dtype:
                            frame = frame.astype(dtype)
                    buf = np.ndarray((h, w, c), dtype=dtype, buffer=self.shms[i].buf)
                    buf[:] = frame
                if not any_alive:
                    break
                # 과도한 CPU 사용 방지. 필요시 조정
                time.sleep(0.05)
        finally:
            for cap in caps:
                cap.release()
            for shm in self.shms:
                if shm is not None:
                    try:
                        shm.close()
                        shm.unlink()
                    except:
                        pass

class IPCamera(Process):
    def __init__(self, paths, stop_event, conn, input_flag):
        super().__init__(daemon=False)

        self.paths = []
        for i, video_path in enumerate(paths):
            self.paths.append(video_path['url'])
        self.stop_event = stop_event
        self.conn = conn
        self.n = len(self.paths)
        self.events = [Event() for _ in range(self.n)]
        self.shms = [None] * self.n
        self.shapes = [None] * self.n
        self.dtypes = [np.uint8] * self.n

    def run(self):

        caps = [cv2.VideoCapture(p) for p in self.paths]
        try:
            # 초기 프레임로 해상도 파악 및 공유메모리 생성
            for i, cap in enumerate(caps):
                ok, frame = cap.read()
                if not ok:
                    self.conn.send(("ERR", i))
                    continue
                h, w, c = frame.shape
                nbytes = frame.nbytes
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self.shms[i] = shm
                self.shapes[i] = (h, w, c)

                # 메타 전송
                self.conn.send(("OK", i, shm.name, (h, w, c), frame.dtype.str))

                # 초기 프레임 한 번 기록
                buf = np.ndarray((h, w, c), dtype=frame.dtype, buffer=shm.buf)
                buf[:] = frame
                self.events[i].set()

            # 카메라 리더 스레드 시작
            for i, cap in enumerate(caps):
                if self.shms[i] is None:
                    continue
                t = threading.Thread(target=self.camera_thread, args=(i, cap), daemon=True)
                t.start()

            # 신호 대기 루프만 유지
            while not self.stop_event.is_set():
                time.sleep(0.01)

        finally:
            self.stop_event.set()
            for cap in caps:
                cap.release()
            # 공유메모리 정리
            for shm in self.shms:
                if shm is not None:
                    try:
                        shm.close()
                        shm.unlink()
                    except:
                        pass

    def camera_thread(self, i, cap):
        h, w, c = self.shapes[i]
        dtype = self.dtypes[i]
        shm = self.shms[i]
        buf = np.ndarray((h, w, c), dtype=dtype, buffer=shm.buf)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            # if frame.shape != (h, w, c) or frame.dtype != dtype:
            #     # 간단화: 해상도 고정. 바뀌면 스킵 또는 리사이즈
            #     frame = cv2.resize(frame, (w, h))
            buf[:] = frame               # 0-copy 덮어쓰기
            self.events[i].set()  
        
