import cv2
import threading
import time
import queue

class IPCamera(threading.Thread):
    def __init__(self, rtsp_cam: list, frame_queue: queue.Queue):
        super().__init__(daemon=True)
        self.rtsp_cam = rtsp_cam
        self.frame_queue = frame_queue
        self.running = True
        self.caps = {}
        
        for cam in self.rtsp_cam:
            cam_id = cam.get('id')
            cam_url = cam.get('url')
            cap = cv2.VideoCapture(cam_url)

            if not cap.isOpened():
                print(f"[ERROR] Failed to open RTSP stream for camera {cam_id} ({cam_url})")
            else:
                print(f"[INFO] Camera {cam_id} connected: {cam_url}")
                self.caps[cam_id] = cap

        self.num_caps = len(self.caps)

    def run(self):
        while self.running:
            frames = []
            for name, cap in self.caps.items():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"[Dataset] Error: Failed to read frame from camera {name}")
                    continue
                frames.append(frame)

            if len(frames) == self.num_caps:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put_nowait(frames)
                except queue.Full:
                    print("[Dataset] Warning: Frame queue still full")


    def stop(self):
        self.running = False
        for cap in self.caps.values():
            cap.release()
