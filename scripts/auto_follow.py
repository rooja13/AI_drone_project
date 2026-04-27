#!/usr/bin/env python3
"""
Autonomous drone following using YOLO + Color Detection with PID controller.

Uses YOLOv8 as primary detector, with color-based detection as fallback.
A PID controller generates velocity commands to keep the vehicle centered.

Usage:
    python3 scripts/auto_follow.py --show-camera
    python3 scripts/auto_follow.py --show-camera --detector hybrid
    python3 scripts/auto_follow.py --show-camera --detector yolo
    python3 scripts/auto_follow.py --show-camera --detector color

Controls:
    q - quit and land
    p - pause/resume
    d - switch detector mode
"""

import time
import argparse
import threading
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.image_pb2 import Image
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "Car", 5: "Bus", 7: "Truck", 3: "Motorcycle"}


class PIDController:
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.reset()

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = max(now - self.last_time, 0.01)
        self.last_time = now
        self.integral = max(-5, min(5, self.integral + error * dt))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.out_min, min(self.out_max, output))


class HybridDetector:
    def __init__(self, mode="hybrid", confidence=0.3):
        self.mode = mode
        self.confidence = confidence
        if mode in ("yolo", "hybrid"):
            print("Loading YOLO model...")
            self.yolo = YOLO("yolov8n.pt")
            print("YOLO loaded!")
        else:
            self.yolo = None

    def detect_yolo(self, frame):
        results = self.yolo(frame, conf=self.confidence, verbose=False)
        best = None
        best_conf = 0
        best_label = ""
        is_vehicle = False

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                if cls_id in VEHICLE_CLASSES:
                    if conf > best_conf or not is_vehicle:
                        best_conf = conf
                        best = (x1, y1, w, h)
                        best_label = VEHICLE_CLASSES[cls_id]
                        is_vehicle = True
                elif not is_vehicle and conf > best_conf:
                    best_conf = conf
                    best = (x1, y1, w, h)
                    best_label = self.yolo.names[cls_id]

        if best:
            x, y, w, h = best
            return True, x + w//2, y + h//2, w, h, (x, y), best_label, best_conf, "YOLO"
        return False, 0, 0, 0, 0, (0, 0), "", 0, "YOLO"

    def detect_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 200:
                x, y, w, h = cv2.boundingRect(largest)
                conf = min(cv2.contourArea(largest) / 5000.0, 1.0)
                return True, x + w//2, y + h//2, w, h, (x, y), "Vehicle", conf, "Color"
        return False, 0, 0, 0, 0, (0, 0), "", 0, "Color"

    def detect(self, frame):
        if self.mode == "yolo":
            return self.detect_yolo(frame)
        elif self.mode == "color":
            return self.detect_color(frame)
        else:
            result = self.detect_yolo(frame)
            if result[0]:
                return result
            return self.detect_color(frame)

    def cycle_mode(self):
        modes = ["hybrid", "yolo", "color"]
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        return self.mode


class AutoFollower:
    def __init__(self, detector_mode="hybrid", show_camera=False):
        self.node = Node()
        self.drone_pub = self.node.advertise("/drone/cmd_vel", Twist)
        self.detector = HybridDetector(mode=detector_mode)
        self.show_camera = show_camera
        self.paused = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.target_box_width = 120
        self.lost_count = 0
        self.max_lost = 30

        self.yaw_pid = PIDController(0.003, 0.0001, 0.001)
        self.fwd_pid = PIDController(0.01, 0.0005, 0.002)
        self.lat_pid = PIDController(0.002, 0.0001, 0.001, -0.5, 0.5)
        self.vert_pid = PIDController(0.002, 0.0001, 0.001, -0.5, 0.5)

        if show_camera:
            cv2.namedWindow("Auto Follow", cv2.WINDOW_NORMAL)

    def image_cb(self, msg: Image):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if arr.size != msg.height * msg.width * 3:
                return
            arr = arr.reshape((msg.height, msg.width, 3))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            with self.frame_lock:
                self.latest_frame = bgr
        except Exception as e:
            print(f"Image error: {e}")

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def send_cmd(self, fwd=0, lat=0, vert=0, yaw=0):
        t = Twist()
        t.linear.x = fwd
        t.linear.y = lat
        t.linear.z = vert
        t.angular.z = yaw
        self.drone_pub.publish(t)

    def stop(self):
        self.send_cmd()

    def draw_hud(self, frame, found, cx, cy, bw, bh, box_xy,
                 label, conf, method, fwd, lat, yaw, vert):
        h, w = frame.shape[:2]
        fcx, fcy = w // 2, h // 2

        if found:
            x, y = box_xy
            color = (0, 255, 0) if method == "YOLO" else (255, 165, 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), color, 2)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), color, 2)
            cv2.line(frame, (fcx, fcy), (cx, cy), (0, 255, 255), 1)
            txt = f"{label} {conf:.0%} [{method}]"
            sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - sz[1] - 10), (x + sz[0] + 5, y), color, -1)
            cv2.putText(frame, txt, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.line(frame, (fcx - 30, fcy), (fcx + 30, fcy), (128, 128, 128), 1)
        cv2.line(frame, (fcx, fcy - 30), (fcx, fcy + 30), (128, 128, 128), 1)

        mode = "PAUSED" if self.paused else ("TRACKING" if found else "SEARCHING")
        mc = (0, 255, 255) if self.paused else ((0, 255, 0) if found else (0, 0, 255))
        cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mc, 2)
        cv2.putText(frame, f"Detector: {self.detector.mode.upper()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Fwd:{fwd:+.2f} Lat:{lat:+.2f} Yaw:{yaw:+.2f} Vert:{vert:+.2f}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def run(self):
        if not self.node.subscribe(Image, "/drone/camera", self.image_cb):
            print("Error subscribing to camera!")
            return

        print("=" * 50)
        print("  AUTONOMOUS DRONE FOLLOWING")
        print("  Detector: " + self.detector.mode.upper())
        print("=" * 50)
        print("  Controls: p=pause  d=switch detector  q=quit")
        print("  Make sure Gazebo is playing + auto_vehicle.py running")
        print("=" * 50)

        print("\nLifting off...")
        for _ in range(50):
            self.send_cmd(vert=0.5)
            time.sleep(0.05)
        print("Airborne! Following...\n")

        try:
            while True:
                frame = self.get_frame()
                fwd = lat = yaw = vert = 0.0
                found = False
                cx = cy = bw = bh = 0
                box_xy = (0, 0)
                label = ""
                conf = 0
                method = ""

                if frame is not None:
                    fh, fw = frame.shape[:2]
                    fcx, fcy = fw // 2, fh // 2

                    found, cx, cy, bw, bh, box_xy, label, conf, method = self.detector.detect(frame)

                    if not self.paused:
                        if found:
                            self.lost_count = 0
                            err_x = cx - fcx
                            err_y = cy - fcy
                            err_dist = self.target_box_width - bw

                            yaw = -self.yaw_pid.compute(err_x)
                            fwd = self.fwd_pid.compute(err_dist)
                            lat = -self.lat_pid.compute(err_x)
                            vert = -self.vert_pid.compute(err_y)

                            self.send_cmd(fwd, lat, vert, yaw)
                            print(f"\rTRACKING [{method}] {label} {conf:.0%} | "
                                  f"fwd={fwd:+.2f} yaw={yaw:+.2f}    ", end="", flush=True)
                        else:
                            self.lost_count += 1
                            if self.lost_count > self.max_lost:
                                self.send_cmd(yaw=0.3)
                                print(f"\rSEARCHING... (rotating)    ", end="", flush=True)
                            else:
                                self.stop()
                                print(f"\rLOST - hovering ({self.lost_count}/{self.max_lost})    ",
                                      end="", flush=True)

                    if self.show_camera:
                        display = self.draw_hud(frame, found, cx, cy, bw, bh,
                                                box_xy, label, conf, method,
                                                fwd, lat, yaw, vert)
                        cv2.imshow("Auto Follow", display)

                if self.show_camera:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("p"):
                        self.paused = not self.paused
                        if self.paused:
                            self.stop()
                        print(f"\n{'PAUSED' if self.paused else 'RESUMED'}")
                    elif key == ord("d"):
                        new_mode = self.detector.cycle_mode()
                        print(f"\nDetector: {new_mode.upper()}")

                time.sleep(0.02)

        except KeyboardInterrupt:
            pass
        finally:
            print("\n\nLanding...")
            for _ in range(40):
                self.send_cmd(vert=-0.3)
                time.sleep(0.05)
            self.stop()
            cv2.destroyAllWindows()
            print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["yolo", "color", "hybrid"],
                        default="hybrid", help="Detection mode (default: hybrid)")
    parser.add_argument("--show-camera", action="store_true", help="Show camera HUD")
    args = parser.parse_args()
    follower = AutoFollower(detector_mode=args.detector, show_camera=args.show_camera)
    follower.run()


if __name__ == "__main__":
    main()
