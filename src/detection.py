import threading
import time

import cv2

import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
FaceAlignerOptions = mp.tasks.vision.FaceAlignerOptions
FaceAligner = mp.tasks.vision.FaceAligner


class DetectionMediaPipe:
    def __init__(self, stop_event, lock, shared_frames, face, log, fps = 30):
        self.stop_event = stop_event
        self.log = log

        self.fps = fps

        # Initialize MediaPipe Face Detection
        self.init_face_detection()

        # Initialize MediaPipe Face Aligner
        self.landmarker_model_path = "models/face_landmarker.task"
        self.init_face_aligner()

        # Initialize MediaPipe Drawing Utils
        self.mp_drawing = mp.solutions.drawing_utils

        self.lock = lock
        self.shared_frames = shared_frames # latest, processed
        self.face = face

    def start(self):
        threading.Thread(target=self.detection_loop, daemon=True).start()

    def detection_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping detection.")
                break

            t1 = time.time()

            with self.lock:
                default_frame = self.shared_frames['default']

            if default_frame is None:
                time.sleep(min(frame_time, 0.01))
                continue

            results = self.detect_face(default_frame)

            # If detections exist
            if results[0].detections is not None:
                detection, bbox = find_largest_detection(default_frame, results[0])
                if bbox is not None:
                    # Align face using largest bbox
                    aligned_face = self.align_face(default_frame, bbox)
                    if aligned_face is not None:
                        # Put aligned face into output
                        with self.lock:
                            self.face['aligned'] = aligned_face

                processed_frame = self.draw_detection(default_frame, detection)
                with self.lock:
                    self.shared_frames['processed'] = processed_frame
            else:
                # Clear processed frame output
                with self.lock:
                    self.shared_frames['processed'] = None

            elapsed_time = time.time() - t1
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)

    def init_face_detection(self):
        mp_face_detection = mp.solutions.face_detection.FaceDetection
        self.face_detection = mp_face_detection(model_selection=0, min_detection_confidence=0.5)

    def init_face_aligner(self):
        with open(self.landmarker_model_path, 'rb') as f:
            model_data = f.read()
        base_options = BaseOptions(model_asset_buffer=model_data)
        options = FaceAlignerOptions(base_options=base_options)
        self.face_aligner = FaceAligner.create_from_options(options)

    # Detect face using MediaPipe Face Detection
    def detect_face(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.face_detection.process(frame)

        return detections

    # Align face using MediaPipe Face Aligner
    def align_face(self, frame, bbox):
        # Extract face ROI
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        # Ensure coordinates are within frame bounds
        x = max(0, int(x))
        y = max(0, int(y))
        x_end = min(frame_w, int(x + w))
        y_end = min(frame_h, int(y + h))

        # Calculate ROI width and height
        roi_w = x_end - x
        roi_h = y_end - y

        # Check for valid ROI
        if roi_w <= 0 or roi_h <= 0:
            self.log.error("Invalid ROI for face alignment.")
            return None

        # Extract the region of interest
        face_roi = frame[y:y + h, x:x + w]

        # Check if the ROI is valid
        if face_roi.size == 0:
            self.log.error("Invalid ROI size for face alignment.")
            return None

        try:
            # Convert to RGB for MediaPipe
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=face_roi_rgb
            )

            # Align face
            aligned_face = self.face_aligner.align(mp_image)

            if aligned_face is not None:
                # Convert back to BGR
                return cv2.cvtColor(aligned_face.numpy_view(), cv2.COLOR_RGB2BGR)
            return None

        except:
            return None

    # Draw single detection on frame
    def draw_detection(self, frame, detection):
        frame.flags.writeable = True

        if frame is not None and detection is not None:
            self.mp_drawing.draw_detection(frame, detection)

        return frame


# Find the largest detection in the results
def find_largest_detection(frame, results):
    h, w, _ = frame.shape
    largest_detection = None
    largest_detection_bbox = None

    if results.detections:
        for detection in results.detections:
            bbox_c = detection.location_data.relative_bounding_box

            # Convert normalized coordinates to pixel values
            x_min = max(0.0, min(1.0, bbox_c.xmin))
            y_min = max(0.0, min(1.0, bbox_c.ymin))
            box_width = max(0.0, min(1.0, bbox_c.width))
            box_height = max(0.0, min(1.0, bbox_c.height))

            # Scale to image size
            x_min = int(x_min * w)
            y_min = int(y_min * h)
            box_width = int(box_width * w)
            box_height = int(box_height * h)

            # Ensure box dimensions are positive
            if box_width > 0 and box_height > 0:
                area = box_width * box_height
                if (largest_detection is None) or (area > largest_detection_bbox[2] * largest_detection_bbox[3]):
                    largest_detection = detection
                    largest_detection_bbox = (x_min, y_min, box_width, box_height)

    return (largest_detection, largest_detection_bbox) if largest_detection is not None else (None, None)
