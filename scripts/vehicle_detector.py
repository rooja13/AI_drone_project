#!/usr/bin/env python3
"""
Detect the red ground vehicle from the drone's camera feed
and draw a bounding box around it.

Uses HSV color-based detection to find the red vehicle.
This serves as a baseline before moving to neural network detection.

Usage:
    python3 scripts/vehicle_detector.py

Controls:
    q - quit
"""

from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
import cv2
import numpy as np
import time
import threading


WINDOW_NAME = "Drone Camera - Vehicle Detection"


class VehicleDetector:
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()
        self.frame_count = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

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

    def detect_red_vehicle(self, frame):
        """
        Detect the red ground vehicle using HSV color filtering.
        Returns the frame with bounding box drawn and detection info.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color has two ranges in HSV (wraps around 0/180)
        # Lower red range
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])

        # Upper red range
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill gaps
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        center_x = 0
        center_y = 0
        box_area = 0

        if contours:
            # Get the largest contour (most likely the vehicle)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Only draw box if the area is large enough (filters out noise)
            if area > 200:
                detected = True
                x, y, w, h = cv2.boundingRect(largest)
                box_area = w * h

                # Calculate center of bounding box
                center_x = x + w // 2
                center_y = y + h // 2

                # Draw bounding box (green)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 6, (0, 255, 0), -1)

                # Draw crosshair lines from center
                cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 0), 2)
                cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 0), 2)

                # Label
                label = f"Vehicle ({w}x{h})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0] + 5, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x + 2, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Show position offset from frame center
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2
                offset_x = center_x - frame_center_x
                offset_y = center_y - frame_center_y

                # Draw frame center crosshair (small, gray)
                cv2.line(frame, (frame_center_x - 20, frame_center_y),
                         (frame_center_x + 20, frame_center_y), (128, 128, 128), 1)
                cv2.line(frame, (frame_center_x, frame_center_y - 20),
                         (frame_center_x, frame_center_y + 20), (128, 128, 128), 1)

                # Show offset info
                cv2.putText(frame, f"Offset: ({offset_x:+d}, {offset_y:+d})",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Status bar at top
        status = "DETECTED" if detected else "SEARCHING..."
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, detected, center_x, center_y, box_area

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()


def main():
    node = Node()
    detector = VehicleDetector()
    topic = "/drone/camera"

    if node.subscribe(Image, topic, detector.image_cb):
        print(f"Subscribed to {topic}")
    else:
        print(f"Error subscribing to {topic}")
        return

    print("Vehicle Detector Running")
    print("Make sure Gazebo simulation is playing!")
    print("Press 'q' to quit\n")

    try:
        while True:
            frame = detector.get_latest_frame()

            if frame is not None:
                # Run detection
                result, detected, cx, cy, area = detector.detect_red_vehicle(frame)

                if detected:
                    print(f"\rDetected at ({cx}, {cy}) area={area}    ", end="", flush=True)

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
