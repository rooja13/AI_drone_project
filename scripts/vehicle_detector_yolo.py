#!/usr/bin/env python3
"""
Detect the ground vehicle from the drone's camera feed using YOLOv8.
Draws a bounding box around detected vehicles.

Uses the pre-trained YOLOv8 model which can detect cars, trucks, etc.
The model runs in real-time on the camera feed from Gazebo.

Usage:
    python3 scripts/vehicle_detector_yolo.py

Controls:
    q - quit

Requirements:
    pip install ultralytics
"""

from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO


WINDOW_NAME = "Drone Camera - YOLO Vehicle Detection"

# COCO class IDs for vehicles
# 2=car, 5=bus, 7=truck, 3=motorcycle, 1=bicycle
VEHICLE_CLASSES = {2: "Car", 5: "Bus", 7: "Truck", 3: "Motorcycle", 1: "Bicycle"}


class YOLOVehicleDetector:
    def __init__(self, model_size="yolov8n.pt", confidence=0.3):
        """
        Initialize YOLO vehicle detector.
        
        Args:
            model_size: YOLO model to use. Options:
                - yolov8n.pt (nano - fastest, least accurate)
                - yolov8s.pt (small - good balance)
                - yolov8m.pt (medium - more accurate)
            confidence: minimum confidence threshold (0.0 to 1.0)
        """
        print(f"Loading YOLO model: {model_size}...")
        self.model = YOLO(model_size)
        self.confidence = confidence
        self.latest_frame = None
        self.lock = threading.Lock()
        self.frame_count = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("YOLO model loaded!")

    def image_cb(self, msg: Image):
        """Callback for camera images from Gazebo."""
        try:
            image_array = np.frombuffer(msg.data, dtype=np.uint8)
            expected_size = msg.height * msg.width * 3
            if image_array.size != expected_size:
                return

            image_array = image_array.reshape((msg.height, msg.width, 3))
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            with self.lock:
                self.latest_frame = image_bgr
                self.frame_count += 1

        except Exception as e:
            print(f"Error processing image: {e}")

    def detect_vehicles(self, frame):
        """
        Run YOLO detection on the frame.
        Returns the annotated frame and detection info.
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence, verbose=False)

        detected = False
        best_box = None
        best_conf = 0
        best_label = ""

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Check if it's a vehicle class OR any object (for simulation)
                if cls_id in VEHICLE_CLASSES:
                    label = VEHICLE_CLASSES[cls_id]
                else:
                    # In Gazebo, the vehicle might not be recognized as a car
                    # so we also accept any detection with good confidence
                    label = self.model.names[cls_id]

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw bounding box
                if cls_id in VEHICLE_CLASSES:
                    # Green for vehicles
                    color = (0, 255, 0)
                    thickness = 3
                else:
                    # Yellow for other objects
                    color = (0, 255, 255)
                    thickness = 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                cv2.line(frame, (center_x - 12, center_y), (center_x + 12, center_y), color, 2)
                cv2.line(frame, (center_x, center_y - 12), (center_x, center_y + 12), color, 2)

                # Label with confidence
                label_text = f"{label} {conf:.0%}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0] + 5, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Track the best vehicle detection
                if cls_id in VEHICLE_CLASSES and conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)
                    best_label = label
                    detected = True
                elif not detected and conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)
                    best_label = label

        # If we have a best detection, show offset from center
        if best_box is not None:
            detected = True
            bx1, by1, bx2, by2 = best_box
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2

            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            offset_x = center_x - frame_center_x
            offset_y = center_y - frame_center_y

            # Draw frame center crosshair
            cv2.line(frame, (frame_center_x - 25, frame_center_y),
                     (frame_center_x + 25, frame_center_y), (128, 128, 128), 1)
            cv2.line(frame, (frame_center_x, frame_center_y - 25),
                     (frame_center_x, frame_center_y + 25), (128, 128, 128), 1)

            # Show offset
            cv2.putText(frame, f"Target: {best_label} | Offset: ({offset_x:+d}, {offset_y:+d})",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Status bar
        status = "TRACKING" if detected else "SEARCHING..."
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "YOLO v8", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, detected

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()


def main():
    node = Node()
    detector = YOLOVehicleDetector(model_size="yolov8n.pt", confidence=0.3)
    topic = "/drone/camera"

    if node.subscribe(Image, topic, detector.image_cb):
        print(f"Subscribed to {topic}")
    else:
        print(f"Error subscribing to {topic}")
        return

    print("\nYOLO Vehicle Detector Running")
    print("Make sure Gazebo simulation is playing!")
    print("Press 'q' to quit\n")

    try:
        while True:
            frame = detector.get_latest_frame()

            if frame is not None:
                result, detected = detector.detect_vehicles(frame)
                cv2.imshow(WINDOW_NAME, result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nDone")


if __name__ == "__main__":
    main()
