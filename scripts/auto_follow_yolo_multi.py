#!/usr/bin/env python3
"""
YOLO-based autonomous following with multiple vehicles.

Uses YOLOv8 to detect ALL vehicles, then lets you select which one
to follow by clicking on it or pressing number keys.

The drone tracks the selected vehicle using position and size matching
across frames (simple tracking).

Usage:
    python3 scripts/auto_follow_yolo_multi.py --show-camera
    python3 scripts/auto_follow_yolo_multi.py --show-camera --model runs/detect/gazebo_vehicle/weights/best.pt

Controls:
    Click on a vehicle in the camera window to select it
    1-9  - select vehicle by detection number
    n    - switch to next vehicle
    p    - pause/resume
    q    - quit and land
"""

import time
import argparse
import threading
import math
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.image_pb2 import Image
from ultralytics import YOLO


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


class VehicleTracker:
    """
    Tracks a specific vehicle across frames using position and size matching.
    When multiple vehicles are detected, it matches the closest one to the
    previously tracked position.
    """

    def __init__(self):
        self.tracked_cx = None
        self.tracked_cy = None
        self.tracked_w = None
        self.tracked_h = None
        self.track_id = None
        self.locked = False

    def select(self, cx, cy, w, h, track_id):
        """Lock onto a specific vehicle."""
        self.tracked_cx = cx
        self.tracked_cy = cy
        self.tracked_w = w
        self.tracked_h = h
        self.track_id = track_id
        self.locked = True

    def clear(self):
        """Clear tracking."""
        self.locked = False
        self.tracked_cx = None
        self.tracked_cy = None
        self.track_id = None

    def find_match(self, detections):
        """
        Find the detection closest to the previously tracked position.
        Returns the index of the best match, or -1 if no good match.
        """
        if not self.locked or self.tracked_cx is None:
            return -1

        best_idx = -1
        best_dist = float('inf')
        max_dist = 200  # max pixel distance to consider a match

        for i, det in enumerate(detections):
            cx, cy = det["cx"], det["cy"]
            dist = math.sqrt((cx - self.tracked_cx) ** 2 + (cy - self.tracked_cy) ** 2)
            if dist < best_dist and dist < max_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            det = detections[best_idx]
            self.tracked_cx = det["cx"]
            self.tracked_cy = det["cy"]
            self.tracked_w = det["w"]
            self.tracked_h = det["h"]

        return best_idx


class YOLOMultiFollower:
    def __init__(self, model_path="runs/detect/gazebo_vehicle/weights/best.pt",
                 show_camera=False, confidence=0.3):
        self.node = Node()
        self.drone_pub = self.node.advertise("/drone/cmd_vel", Twist)
        self.show_camera = show_camera
        self.paused = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.target_box_width = 120
        self.lost_count = 0
        self.max_lost = 30

        # YOLO model
        print(f"Loading YOLO model: {model_path}...")
        self.yolo = YOLO(model_path)
        self.confidence = confidence
        print("YOLO loaded!")

        # Vehicle tracker
        self.tracker = VehicleTracker()

        # Mouse click position
        self.click_x = -1
        self.click_y = -1

        # PID controllers
        self.yaw_pid = PIDController(0.003, 0.0001, 0.001)
        self.fwd_pid = PIDController(0.01, 0.0005, 0.002)
        self.lat_pid = PIDController(0.002, 0.0001, 0.001, -0.5, 0.5)
        self.vert_pid = PIDController(0.002, 0.0001, 0.001, -0.5, 0.5)

        if show_camera:
            cv2.namedWindow("YOLO Multi-Vehicle Follow", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("YOLO Multi-Vehicle Follow", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select a vehicle."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_x = x
            self.click_y = y

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

    def detect_vehicles(self, frame):
        """Run YOLO and return all detections."""
        results = self.yolo(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                label = self.yolo.names[cls_id]

                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "w": w, "h": h, "cx": cx, "cy": cy,
                    "conf": conf, "label": label, "cls_id": cls_id
                })

        return detections

    def check_click_selection(self, detections):
        """Check if a mouse click landed on a vehicle."""
        if self.click_x < 0:
            return

        for i, det in enumerate(detections):
            if (det["x1"] <= self.click_x <= det["x2"] and
                    det["y1"] <= self.click_y <= det["y2"]):
                self.tracker.select(det["cx"], det["cy"], det["w"], det["h"], i)
                print(f"\nSelected vehicle #{i + 1}: {det['label']} ({det['conf']:.0%})")
                self.yaw_pid.reset()
                self.fwd_pid.reset()
                break

        self.click_x = -1
        self.click_y = -1

    def draw_hud(self, frame, detections, tracked_idx, fwd, lat, yaw, vert):
        h, w = frame.shape[:2]
        fcx, fcy = w // 2, h // 2

        # Draw all detections
        for i, det in enumerate(detections):
            is_tracked = (i == tracked_idx)

            if is_tracked:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (0, 255, 255)
                thickness = 1

            # Bounding box
            cv2.rectangle(frame, (det["x1"], det["y1"]),
                          (det["x2"], det["y2"]), color, thickness)

            # Label
            if is_tracked:
                tag = f">> #{i + 1} {det['label']} {det['conf']:.0%} <<"
            else:
                tag = f"#{i + 1} {det['label']} {det['conf']:.0%}"

            sz = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (det["x1"], det["y1"] - sz[1] - 8),
                          (det["x1"] + sz[0] + 4, det["y1"]), color, -1)
            cv2.putText(frame, tag, (det["x1"] + 2, det["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Crosshair on tracked target
            if is_tracked:
                cv2.circle(frame, (det["cx"], det["cy"]), 6, color, -1)
                cv2.line(frame, (det["cx"] - 15, det["cy"]),
                         (det["cx"] + 15, det["cy"]), color, 2)
                cv2.line(frame, (det["cx"], det["cy"] - 15),
                         (det["cx"], det["cy"] + 15), color, 2)
                cv2.line(frame, (fcx, fcy), (det["cx"], det["cy"]), (0, 255, 255), 1)

        # Frame center crosshair
        cv2.line(frame, (fcx - 30, fcy), (fcx + 30, fcy), (128, 128, 128), 1)
        cv2.line(frame, (fcx, fcy - 30), (fcx, fcy + 30), (128, 128, 128), 1)

        # Status
        found = tracked_idx >= 0
        mode = "PAUSED" if self.paused else ("TRACKING" if found else
                ("SEARCHING" if self.tracker.locked else "SELECT A VEHICLE"))
        mc = (0, 255, 255) if self.paused else ((0, 255, 0) if found else (0, 0, 255))
        cv2.putText(frame, f"Mode: {mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mc, 2)

        # YOLO badge
        cv2.putText(frame, "YOLOv8", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Vehicle count
        cv2.putText(frame, f"Vehicles: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Tracked vehicle info
        if tracked_idx >= 0:
            det = detections[tracked_idx]
            cv2.putText(frame, f"Following: #{tracked_idx + 1} {det['label']}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Controls
        cv2.putText(frame, "Click vehicle to select | n=next | p=pause | q=quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Commands
        cv2.putText(frame, f"Fwd:{fwd:+.2f} Lat:{lat:+.2f} Yaw:{yaw:+.2f} Vert:{vert:+.2f}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame

    def run(self):
        if not self.node.subscribe(Image, "/drone/camera", self.image_cb):
            print("Error subscribing to camera!")
            return

        print("=" * 55)
        print("  YOLO MULTI-VEHICLE AUTONOMOUS FOLLOWING")
        print("=" * 55)
        print("  Click on a vehicle in the camera to follow it")
        print("  Or press 1-9 to select by number")
        print("  n = next vehicle | p = pause | q = quit")
        print("=" * 55)

        print("\nLifting off...")
        for _ in range(50):
            self.send_cmd(vert=0.5)
            time.sleep(0.05)
        print("Airborne!\n")

        try:
            while True:
                frame = self.get_frame()
                fwd = lat = yaw = vert = 0.0
                tracked_idx = -1

                if frame is not None:
                    fh, fw = frame.shape[:2]
                    fcx, fcy = fw // 2, fh // 2

                    # Detect all vehicles
                    detections = self.detect_vehicles(frame)

                    # Check for click selection
                    self.check_click_selection(detections)

                    # Find tracked vehicle
                    if self.tracker.locked:
                        tracked_idx = self.tracker.find_match(detections)

                    # Auto-select first vehicle if none selected
                    if not self.tracker.locked and len(detections) > 0:
                        print("\nAuto-selecting vehicle #1. Click another to switch.")
                        det = detections[0]
                        self.tracker.select(det["cx"], det["cy"], det["w"], det["h"], 0)
                        tracked_idx = 0

                    if not self.paused:
                        if tracked_idx >= 0:
                            self.lost_count = 0
                            det = detections[tracked_idx]
                            cx, cy = det["cx"], det["cy"]
                            bw = det["w"]

                            err_x = cx - fcx
                            err_y = cy - fcy
                            err_dist = self.target_box_width - bw

                            yaw = -self.yaw_pid.compute(err_x)
                            fwd = self.fwd_pid.compute(err_dist)
                            lat = -self.lat_pid.compute(err_x)
                            vert = -self.vert_pid.compute(err_y)

                            self.send_cmd(fwd, lat, vert, yaw)
                            print(f"\rTRACKING #{tracked_idx + 1} {det['label']} {det['conf']:.0%} | "
                                  f"fwd={fwd:+.2f} yaw={yaw:+.2f} | "
                                  f"Vehicles: {len(detections)}    ", end="", flush=True)
                        elif self.tracker.locked:
                            self.lost_count += 1
                            if self.lost_count > self.max_lost:
                                self.send_cmd(yaw=0.3)
                                print(f"\rSEARCHING...    ", end="", flush=True)
                            else:
                                self.stop()

                    if self.show_camera:
                        display = self.draw_hud(frame, detections, tracked_idx,
                                                fwd, lat, yaw, vert)
                        cv2.imshow("YOLO Multi-Vehicle Follow", display)

                if self.show_camera:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("p"):
                        self.paused = not self.paused
                        if self.paused:
                            self.stop()
                        print(f"\n{'PAUSED' if self.paused else 'RESUMED'}")
                    elif key == ord("n"):
                        # Switch to next vehicle
                        if detections:
                            next_idx = 0
                            if tracked_idx >= 0:
                                next_idx = (tracked_idx + 1) % len(detections)
                            det = detections[next_idx]
                            self.tracker.select(det["cx"], det["cy"], det["w"], det["h"], next_idx)
                            self.yaw_pid.reset()
                            self.fwd_pid.reset()
                            print(f"\nSwitched to vehicle #{next_idx + 1}")
                    elif ord("1") <= key <= ord("9"):
                        idx = key - ord("1")
                        if idx < len(detections):
                            det = detections[idx]
                            self.tracker.select(det["cx"], det["cy"], det["w"], det["h"], idx)
                            self.yaw_pid.reset()
                            self.fwd_pid.reset()
                            print(f"\nSelected vehicle #{idx + 1}")

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
    parser.add_argument("--model", default="runs/detect/gazebo_vehicle/weights/best.pt",
                        help="Path to YOLO model")
    parser.add_argument("--show-camera", action="store_true", help="Show camera HUD")
    parser.add_argument("--confidence", type=float, default=0.3, help="YOLO confidence threshold")
    args = parser.parse_args()

    follower = YOLOMultiFollower(
        model_path=args.model,
        show_camera=args.show_camera,
        confidence=args.confidence
    )
    follower.run()


if __name__ == "__main__":
    main()
