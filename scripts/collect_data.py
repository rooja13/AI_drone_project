#!/usr/bin/env python3
"""
Collect training images from Gazebo drone camera.
Saves frames to a folder for YOLO training.

Usage:
    python3 scripts/collect_data.py

Controls:
    s - save current frame
    a - toggle auto-save (saves every 10 frames)
    q - quit
"""

import os
import time
import threading
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image


class DataCollector:
    def __init__(self, output_dir="training_data/images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.count = 0
        self.auto_save = False
        self.frame_num = 0
        cv2.namedWindow("Data Collector", cv2.WINDOW_NORMAL)

    def image_cb(self, msg: Image):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if arr.size != msg.height * msg.width * 3:
                return
            arr = arr.reshape((msg.height, msg.width, 3))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            with self.lock:
                self.latest_frame = bgr
                self.frame_num += 1
        except Exception as e:
            print(f"Error: {e}")

    def save_frame(self, frame):
        filename = os.path.join(self.output_dir, f"frame_{self.count:05d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        self.count += 1

    def run(self):
        node = Node()
        if not node.subscribe(Image, "/drone/camera", self.image_cb):
            print("Error subscribing!")
            return

        print("Data Collector Running")
        print("s = save frame | a = toggle auto-save | q = quit")
        print(f"Saving to: {self.output_dir}\n")

        try:
            while True:
                with self.lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None

                if frame is not None:
                    display = frame.copy()
                    status = "AUTO-SAVE ON" if self.auto_save else "Manual"
                    cv2.putText(display, f"Mode: {status} | Saved: {self.count}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Data Collector", display)

                    if self.auto_save and self.frame_num % 10 == 0:
                        self.save_frame(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    if frame is not None:
                        self.save_frame(frame)
                elif key == ord("a"):
                    self.auto_save = not self.auto_save
                    print(f"Auto-save: {'ON' if self.auto_save else 'OFF'}")

                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            print(f"\nDone! Collected {self.count} images")


if __name__ == "__main__":
    DataCollector().run()
