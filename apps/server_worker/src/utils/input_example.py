import os.path as osp
import glob
import cv2
import numpy as np
import threading
import queue

class ExampleDataset(threading.Thread):
    """
    Example Dataset class for FOCUS Inference with parallel processing and thread-safe queueing.
    """
    def __init__(self, cfg, example_path=None, start_idx=0, end_idx=None, frame_queue=None):
        super().__init__(daemon=True)
        self.example_path = example_path
        self.video_paths = sorted(glob.glob(osp.join(self.example_path, 'hdVideos', '*.mp4')))
        
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.frame_queue = frame_queue 
        self.running = True
        
        self.caps = {}
        for i, path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_idx)
            self.caps[i] = cap

    def run(self):
        while self.running:
            frames = []

            for i in range(len(self.caps)):
                ret, frame = self.caps[i].read()
                if not ret:
                    print(f"[Dataset] Error: Failed to read frame from camera {i}")
                    break  # 프레임이 하나라도 없으면 skip
                frames.append(frame)

            if len(frames) == len(self.caps):
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # 오래된 프레임 제거
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(frames)
                except queue.Full:
                    print("[Dataset] Warning: Frame queue is full")

    def stop(self):
        self.running = False
        for cap in self.caps.values():
            cap.release()