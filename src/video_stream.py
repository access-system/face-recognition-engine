import time

import cv2


class VideoStream:
    def __init__(self, stop_event, lock, shared_frames, shared_access, log, name: str = "Face Detection", fps = 30):
        self.stop_event = stop_event
        self.log = log

        self.name = name
        self.fps = fps

        self.lock = lock
        self.shared_frames = shared_frames
        self.shared_access = shared_access

        self.access_statuses = {
            0: ("Access Granted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2),
            1: ("Access Denied", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2),
            2: ("Not Found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2),
        }

    def start(self):
        self.stream_loop()

    def stream_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            # Check for stop event
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping video stream.")
                break

            t1 = time.time()

            with self.lock:
                processed_frame = self.shared_frames['processed']
                default_frame = self.shared_frames['default']

                access_status = self.shared_access['status']

            if default_frame is None:
                continue

            if access_status is None:
                access_status = 2

            # Display processed frame if available, else display latest frame
            if (processed_frame is not None) and (access_status == 0 or access_status == 1):
                frame = cv2.flip(processed_frame, 1)
                cv2.putText(frame, *self.access_statuses[access_status])
            else:
                frame = cv2.flip(default_frame, 1)
                cv2.putText(frame, *self.access_statuses[2])

            cv2.imshow(self.name, frame)


            elapsed_time = time.time() - t1
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)

            if cv2.waitKey(5) & 0xFF == 27:
                self.stop_event.set()

        cv2.destroyAllWindows()
