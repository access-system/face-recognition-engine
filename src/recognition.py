import threading
import time

import openvino as ov
import numpy as np
import cv2


class RecognitionArcFace:
    def __init__(self, stop_event, lock, face, shared_embedding, log, device = 'CPU', fps = 30):
        self.stop_event = stop_event
        self.log = log

        self.fps = fps

        self.core = ov.Core()
        self.device = device
        self.init_arcface()

        self.lock = lock
        self.face = face
        self.shared_embedding = shared_embedding

    def init_arcface(self):
        self.arcface_resnet100_model_path = "models/arcfaceresnet100-8.onnx"
        self.arcface_resnet100_model = self.core.read_model(model=self.arcface_resnet100_model_path)
        self.arcface_resnet100_compiled = self.core.compile_model(model=self.arcface_resnet100_model,
                                                                  device_name=self.device)

    def start(self):
        threading.Thread(target=self.recognition_loop, daemon=True).start()

    def recognition_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping recognition.")
                break

            t1 = time.time()

            with self.lock:
                aligned_face = self.face['aligned']

            if aligned_face is None:
                with self.lock:
                    self.shared_embedding['default'] = None

                time.sleep(min(frame_time, 0.01))
                continue

            embedding = self.recognize(aligned_face)

            normalized_embedding = l2_norm(embedding)
            with self.lock:
                self.shared_embedding['default'] = normalized_embedding

            elapsed_time = time.time() - t1
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)

    def recognize(self, face_img):
        input_data = preprocess_arcface(face_img)
        output_layer = self.arcface_resnet100_compiled.output(0)

        embeddings = self.arcface_resnet100_compiled([input_data])[output_layer]

        return embeddings[0]

    def __str__(self):
        return f"Available devices: {self.core.available_devices}"


# Preprocess frame for ArcFace model
def preprocess_arcface(frame):
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize
    frame = cv2.resize(frame, (112, 112))
    # Scale to [0, 1]
    frame = frame.astype(np.float32)
    # Change data layout from HWC to CHW
    frame = np.transpose(frame, (2, 0, 1))
    # Add batch dimension
    frame = np.expand_dims(frame, 0)

    return frame


# Postprocess ArcFace embedding with L2 normalization
def l2_norm(embedding: np.ndarray):
    norm = np.linalg.norm(embedding)
    normalized_embedding = embedding / norm

    return normalized_embedding
