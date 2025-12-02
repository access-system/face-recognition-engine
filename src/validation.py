import threading
import time

from api.access_system import validate_embedding


class EmbeddingValidation:
    def __init__(self, stop_event, cache, lock, shared_embedding, shared_access, log, fps = 30):
        self.stop_event = stop_event
        self.log = log
        self.cache = cache

        self.fps = fps

        self.lock = lock
        self.shared_embedding = shared_embedding
        self.shared_access = shared_access

    def start(self):
        threading.Thread(target=self.validation_loop, daemon=True).start()

    def validation_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping validation.")
                break

            t1 = time.time()

            with self.lock:
                shared_embedding = self.shared_embedding['default']

            if shared_embedding is None:
                with self.lock:
                    self.shared_access['status'] = 2  # Not Found
                time.sleep(min(frame_time, 0.01))
                continue

            # Verify embedding against cache
            # self.log.info("Validating embedding...")
            if self.cache.verify_embedding(shared_embedding):
                # self.log.info("Embedding found in cache.")
                with self.lock:
                    self.shared_access['status'] = 0  # Access Granted
                time.sleep(1)
            else:
                # If not found in cache, validate via API
                exists, msg = validate_embedding(shared_embedding)

                if exists:
                    # self.log.info("Embedding found on server.")
                    self.cache.store_embedding(shared_embedding)

                    with self.lock:
                        self.shared_access['status'] = 0  # Access Granted
                    time.sleep(1)
                else:
                    # self.log.info("Embedding not found.")
                    with self.lock:
                        self.shared_access['status'] = 1  # Access Denied
                    time.sleep(1)


            elapsed_time = time.time() - t1
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)
