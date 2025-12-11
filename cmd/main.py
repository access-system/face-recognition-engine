import threading

import loguru

from src.cache import VerifiedEmbeddingsCache
from src.video_capture import VideoCapture
from src.video_stream import VideoStream
from src.detection import DetectionMediaPipe
from src.recognition import RecognitionArcFace
from src.validation import EmbeddingValidation


def main():
    lock = threading.Lock()

    # Event to signal threads to stop
    stop_event = threading.Event()

    log = loguru.logger

    shared_frames = {'default': None, 'processed': None}
    face = {'aligned': None}
    shared_embedding = {'default': None}
    shared_access = {'status': None}
    shared_face_data = {'name': None, 'accuracy': None}

    log.info("Initializing cache...")
    cache = VerifiedEmbeddingsCache(log, 'localhost', 6379, 0)

    fps = 30
    log.info(f"Set FPS to {fps}...")

    main_camera = 0

    log.info("Setup pipelines...")
    video_capture = VideoCapture(stop_event, lock, shared_frames, log, fps=fps, device=main_camera)
    video_stream = VideoStream(stop_event, lock, shared_frames, shared_access, shared_face_data, log, fps=fps)
    detection_mediapipe = DetectionMediaPipe(stop_event, lock, shared_frames, face, log, fps=fps)
    recognition_arcface = RecognitionArcFace(stop_event, lock, face, shared_embedding, log, device='GPU',
                                             fps=fps)
    embedding_validation = EmbeddingValidation(stop_event, cache, lock, shared_embedding, shared_access, shared_face_data, log, fps=fps)

    log.info("Starting pipelines...")
    video_capture.start()
    detection_mediapipe.start()
    recognition_arcface.start()
    embedding_validation.start()

    log.info("Starting video stream...")
    video_stream.start()


if __name__ == "__main__":
    main()
