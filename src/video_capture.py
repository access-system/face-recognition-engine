import threading
import time

import cv2


class VideoCapture:
    def __init__(self, stop_event, lock, shared_frames, log, fps = 30, device = 0):
        self.stop_event = stop_event
        self.log = log

        self.cap = cv2.VideoCapture(device)
        self.fps = fps

        self.lock = lock
        self.shared_frames = shared_frames

    def start(self):
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def stop(self):
        self.stop_event.set()

    def capture_loop(self):
        frame_time = 1.0 / self.fps

        while self.cap.isOpened():
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping video capture.")
                break

            ret, frame = self.cap.read()
            if not ret:
                continue

            t1 = time.time()

            # Put frame into output
            with self.lock:
                self.shared_frames['default'] = frame

            elapsed_time = time.time() - t1
            # self.log.info(f"{elapsed_time:.3f} seconds per frame")
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)
