#!/usr/bin/env python3

from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
import cv2
import numpy as np
import time
import argparse
import threading

WINDOW_NAME = "Sim Drone Camera"


class CameraFeed:
    def __init__(self, show=False):
        self.show = show
        self.frame_count = 0
        self.latest_frame = None
        self.lock = threading.Lock()
        self.last_frame_time = None

        if self.show:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def image_cb(self, msg: Image):
        try:
            image_array = np.frombuffer(msg.data, dtype=np.uint8)

            expected_size = msg.height * msg.width * 3
            if image_array.size != expected_size:
                print(f"Bad frame size: got {image_array.size}, expected {expected_size}")
                return

            image_array = image_array.reshape((msg.height, msg.width, 3))
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            with self.lock:
                self.latest_frame = image_bgr
                self.frame_count += 1
                self.last_frame_time = time.time()

        except Exception as e:
            print(f"Error processing image: {e}")

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.frame_count, self.last_frame_time
            return self.latest_frame.copy(), self.frame_count, self.last_frame_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Display live camera feed")
    args = parser.parse_args()

    node = Node()
    camera = CameraFeed(show=args.show)
    topic_image_msg = "/drone/camera"

    if node.subscribe(Image, topic_image_msg, camera.image_cb):
        print(f"Subscribed to {topic_image_msg}")
        print(f"Live display: {'ON' if args.show else 'OFF'}")
    else:
        print(f"Error subscribing to topic {topic_image_msg}")
        return

    last_report = time.time()

    try:
        while True:
            frame, frame_count, last_frame_time = camera.get_latest_frame()

            if args.show and frame is not None:
                display_frame = frame

                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            now = time.time()
            if now - last_report > 2.0:
                if last_frame_time is None:
                    print("No frames received yet...")
                else:
                    age = now - last_frame_time
                    print(f"Frames received: {frame_count}, last frame age: {age:.2f}s")
                last_report = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
