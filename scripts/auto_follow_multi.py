#!/usr/bin/env python3
"""
Autonomous following with multiple vehicles - follow a specific colored vehicle.

The drone can choose which vehicle to follow based on color:
  - Red vehicle (default)
  - Blue vehicle
  - Green vehicle

Usage:
    python3 scripts/auto_follow_multi.py --show-camera --target red
    python3 scripts/auto_follow_multi.py --show-camera --target blue
    python3 scripts/auto_follow_multi.py --show-camera --target green

Controls:
    q - quit and land
    p - pause/resume
    1 - switch to follow RED vehicle
    2 - switch to follow BLUE vehicle
    3 - switch to follow GREEN vehicle
"""

import time
import argparse
import threading
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.image_pb2 import Image


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


class ColorTargetDetector:
    """Detect vehicles by their specific color."""

    # HSV color ranges for each vehicle
    COLOR_RANGES = {
        "red": {
            "ranges": [
                (np.array([0, 80, 80]), np.array([10, 255, 255])),
                (np.array([160, 80, 80]), np.array([180, 255, 255]))
            ],
            "display_color": (0, 0, 255),   # BGR for drawing
            "label": "RED Vehicle"
        },
        "blue": {
            "ranges": [
                (np.array([100, 80, 80]), np.array([130, 255, 255]))
            ],
            "display_color": (255, 0, 0),
            "label": "BLUE Vehicle"
        },
        "green": {
            "ranges": [
                (np.array([35, 80, 80]), np.array([85, 255, 255]))
            ],
            "display_color": (0, 255, 0),
            "label": "GREEN Vehicle"
        }
    }

    def __init__(self, target="red"):
        self.target = target

    def set_target(self, target):
        self.target = target

    def detect_all(self, frame):
        """Detect all colored vehicles and return their positions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        detections = {}

        for color_name, config in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in config["ranges"]:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 200:
                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    cy = y + h // 2
                    detections[color_name] = {
                        "cx": cx, "cy": cy, "w": w, "h": h,
                        "x": x, "y": y, "area": area,
                        "color": config["display_color"],
                        "label": config["label"]
                    }

        return detections

    def detect_target(self, frame):
        """Detect only the target vehicle."""
        detections = self.detect_all(frame)
        if self.target in detections:
            d = detections[self.target]
            return True, d["cx"], d["cy"], d["w"], d["h"], (d["x"], d["y"]), d["label"], detections
        return False, 0, 0, 0, 0, (0, 0), "", detections


class MultiVehicleFollower:
    def __init__(self, target="red", show_camera=False):
        self.node = Node()
        self.drone_pub = self.node.advertise("/drone/cmd_vel", Twist)
        self.detector = ColorTargetDetector(target=target)
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
            cv2.namedWindow("Multi-Vehicle Follow", cv2.WINDOW_NORMAL)

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
                 label, detections, fwd, lat, yaw, vert):
        h, w = frame.shape[:2]
        fcx, fcy = w // 2, h // 2

        # Draw ALL detected vehicles (non-target with thin box)
        for color_name, d in detections.items():
            is_target = (color_name == self.detector.target)
            thickness = 3 if is_target else 1
            box_color = d["color"]

            cv2.rectangle(frame, (d["x"], d["y"]),
                          (d["x"] + d["w"], d["y"] + d["h"]), box_color, thickness)

            tag = f">> {d['label']} <<" if is_target else d["label"]
            cv2.putText(frame, tag, (d["x"], d["y"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2 if is_target else 1)

            if is_target:
                cv2.circle(frame, (d["cx"], d["cy"]), 6, box_color, -1)
                cv2.line(frame, (d["cx"] - 15, d["cy"]), (d["cx"] + 15, d["cy"]), box_color, 2)
                cv2.line(frame, (d["cx"], d["cy"] - 15), (d["cx"], d["cy"] + 15), box_color, 2)
                cv2.line(frame, (fcx, fcy), (d["cx"], d["cy"]), (0, 255, 255), 1)

        # Frame center crosshair
        cv2.line(frame, (fcx - 30, fcy), (fcx + 30, fcy), (128, 128, 128), 1)
        cv2.line(frame, (fcx, fcy - 30), (fcx, fcy + 30), (128, 128, 128), 1)

        # Status
        mode = "PAUSED" if self.paused else ("TRACKING" if found else "SEARCHING")
        mc = (0, 255, 255) if self.paused else ((0, 255, 0) if found else (0, 0, 255))
        cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mc, 2)

        # Target indicator
        target_color = self.detector.COLOR_RANGES[self.detector.target]["display_color"]
        cv2.putText(frame, f"Target: {self.detector.target.upper()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)

        # Vehicle count
        cv2.putText(frame, f"Vehicles detected: {len(detections)}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls reminder
        cv2.putText(frame, "1=Red 2=Blue 3=Green p=Pause q=Quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Command values
        cv2.putText(frame, f"Fwd:{fwd:+.2f} Lat:{lat:+.2f} Yaw:{yaw:+.2f} Vert:{vert:+.2f}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame

    def run(self):
        if not self.node.subscribe(Image, "/drone/camera", self.image_cb):
            print("Error subscribing to camera!")
            return

        print("=" * 50)
        print("  MULTI-VEHICLE AUTONOMOUS FOLLOWING")
        print(f"  Target: {self.detector.target.upper()}")
        print("=" * 50)
        print("  Controls:")
        print("  1 = Follow RED    2 = Follow BLUE")
        print("  3 = Follow GREEN  p = Pause  q = Quit")
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
                detections = {}

                if frame is not None:
                    fh, fw = frame.shape[:2]
                    fcx, fcy = fw // 2, fh // 2

                    found, cx, cy, bw, bh, box_xy, label, detections = self.detector.detect_target(frame)

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
                            print(f"\rTRACKING {label} | "
                                  f"fwd={fwd:+.2f} yaw={yaw:+.2f} | "
                                  f"Vehicles: {len(detections)}    ", end="", flush=True)
                        else:
                            self.lost_count += 1
                            if self.lost_count > self.max_lost:
                                self.send_cmd(yaw=0.3)
                                print(f"\rSEARCHING for {self.detector.target.upper()}...    ",
                                      end="", flush=True)
                            else:
                                self.stop()

                    if self.show_camera:
                        display = self.draw_hud(frame, found, cx, cy, bw, bh,
                                                box_xy, label, detections,
                                                fwd, lat, yaw, vert)
                        cv2.imshow("Multi-Vehicle Follow", display)

                if self.show_camera:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("p"):
                        self.paused = not self.paused
                        if self.paused:
                            self.stop()
                        print(f"\n{'PAUSED' if self.paused else 'RESUMED'}")
                    elif key == ord("1"):
                        self.detector.set_target("red")
                        self.yaw_pid.reset()
                        self.fwd_pid.reset()
                        print(f"\nSwitched to RED vehicle")
                    elif key == ord("2"):
                        self.detector.set_target("blue")
                        self.yaw_pid.reset()
                        self.fwd_pid.reset()
                        print(f"\nSwitched to BLUE vehicle")
                    elif key == ord("3"):
                        self.detector.set_target("green")
                        self.yaw_pid.reset()
                        self.fwd_pid.reset()
                        print(f"\nSwitched to GREEN vehicle")

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
    parser.add_argument("--target", choices=["red", "blue", "green"],
                        default="red", help="Which vehicle to follow (default: red)")
    parser.add_argument("--show-camera", action="store_true", help="Show camera HUD")
    args = parser.parse_args()
    follower = MultiVehicleFollower(target=args.target, show_camera=args.show_camera)
    follower.run()


if __name__ == "__main__":
    main()
