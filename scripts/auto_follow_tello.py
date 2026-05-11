#!/usr/bin/env python3
"""
Autonomous following on the real DJI Tello drone.

Uses the Tello's camera feed + custom YOLO model to detect and follow
a ground vehicle in real life.

Architecture:
    Tello Camera → YOLO Detection → PID Controller → MotionCommand → UDP → command_conversion → tello_bridge → Tello

Setup:
    1. Connect to Tello's Wi-Fi
    2. Terminal 1: python3 scripts/tello_bridge.py --rc-port 6005 --takeoff --show-video
    3. Terminal 2: python3 scripts/command_conversion.py --input_port 5005 --output_port 6005
    4. Terminal 3: python3 scripts/auto_follow_tello.py

Controls:
    q - quit (sends stop command)
    p - pause/resume following
    d - switch detector mode (yolo/color/hybrid)

Requirements:
    pip install ultralytics opencv-python numpy djitellopy
"""

import time
import argparse
import json
import socket
import threading
import numpy as np
import cv2
from djitellopy import Tello
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


class HybridDetector:
    def __init__(self, mode="hybrid", model_path="runs/detect/gazebo_vehicle/weights/best.pt",
                 confidence=0.55, color_min_area=1000):
        self.mode = mode
        self.confidence = confidence
        self.color_min_area = color_min_area
        if mode in ("yolo", "hybrid"):
            print(f"Loading YOLO model: {model_path}...")
            self.yolo = YOLO(model_path)
            print("YOLO loaded!")
        else:
            self.yolo = None

    def detect_yolo(self, frame):
        results = self.yolo(frame, conf=self.confidence, verbose=False)
        best = None
        best_conf = 0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if conf > best_conf:
                    best_conf = conf
                    best = (x1, y1, w, h)

        if best:
            x, y, w, h = best
            return True, x + w // 2, y + h // 2, w, h, (x, y), "Vehicle", best_conf, "YOLO"
        return False, 0, 0, 0, 0, (0, 0), "", 0, "YOLO"

    def detect_color(self, frame):
        # Frames passed into the detector are RGB. Keep the color fallback
        # consistent with the YOLO input path.
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
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
            area = cv2.contourArea(largest)
            if area > self.color_min_area:
                x, y, w, h = cv2.boundingRect(largest)
                conf = min(area / 5000.0, 1.0)
                return True, x + w // 2, y + h // 2, w, h, (x, y), "Vehicle", conf, "Color"
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


class TelloAutoFollow:
    """Autonomous following using the Tello's own camera."""

    def __init__(self, detector_mode="hybrid", model_path="runs/detect/gazebo_vehicle/weights/best.pt",
                 use_udp=False, udp_port=5005, detection_confidence=0.55,
                 target_y_ratio=0.75):
        self.detector = HybridDetector(mode=detector_mode, model_path=model_path,
                                       confidence=detection_confidence)
        self.paused = False
        self.use_udp = use_udp
        self.target_box_width = 150  # larger for real world
        # Aim to keep the detected ground robot centered left/right and in the
        # lower quarter of the camera image vertically. 0.75 means 75% down
        # from the top of the frame.
        self.target_y_ratio = target_y_ratio
        self.lost_count = 0
        self.max_lost = 30

        # PID controllers (tuned for real Tello - less aggressive than sim)
        self.yaw_pid = PIDController(0.002, 0.00005, 0.0008)
        self.fwd_pid = PIDController(0.008, 0.0003, 0.001)
        self.lat_pid = PIDController(0.0015, 0.00005, 0.0005, -0.4, 0.4)
        self.vert_pid = PIDController(0.0015, 0.00005, 0.0005, -0.4, 0.4)

        if use_udp:
            # Send commands via UDP to command_conversion.py
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_addr = ("127.0.0.1", udp_port)
            self.tello = None
            print(f"Sending commands via UDP to port {udp_port}")
        else:
            # Connect directly to Tello
            self.tello = Tello()
            self.sock = None
            print("Connecting directly to Tello...")

        cv2.namedWindow("Tello Auto Follow", cv2.WINDOW_NORMAL)

    def send_cmd(self, fwd=0, right=0, up=0, yaw=0):
        if self.use_udp:
            cmd = {
                "forward": max(-1, min(1, fwd)),
                "right": max(-1, min(1, right)),
                "up": max(-1, min(1, up)),
                "yaw": max(-1, min(1, yaw))
            }
            payload = json.dumps(cmd).encode("utf-8")
            self.sock.sendto(payload, self.udp_addr)
        else:
            # Direct Tello RC command (values -100 to 100)
            lr = int(max(-100, min(100, right * 100)))
            fb = int(max(-100, min(100, fwd * 100)))
            ud = int(max(-100, min(100, up * 100)))
            yw = int(max(-100, min(100, yaw * 100)))
            self.tello.send_rc_control(lr, fb, ud, yw)

    def stop(self):
        self.send_cmd(0, 0, 0, 0)

    def draw_hud(self, frame, found, cx, cy, bw, bh, box_xy,
                 label, conf, method, fwd, lat, yaw, vert):
        h, w = frame.shape[:2]
        fcx = w // 2
        fcy = int(h * self.target_y_ratio)

        if found:
            x, y = box_xy
            # draw_hud receives a BGR display frame, so color tuples are BGR.
            color = (0, 255, 0) if method == "YOLO" else (0, 165, 255)
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
        cv2.putText(frame, "Target", (fcx + 10, fcy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        mode = "PAUSED" if self.paused else ("TRACKING" if found else "SEARCHING")
        mc = (0, 255, 255) if self.paused else ((0, 255, 0) if found else (0, 0, 255))
        cv2.putText(frame, f"TELLO | Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mc, 2)
        cv2.putText(frame, f"Detector: {self.detector.mode.upper()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.tello:
            try:
                battery = self.tello.get_battery()
                if battery is not None:
                    cv2.putText(frame, f"Battery: {battery}%", (w - 180, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception:
                pass

        cv2.putText(frame, f"Fwd:{fwd:+.2f} Lat:{lat:+.2f} Yaw:{yaw:+.2f} Vert:{vert:+.2f}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def run(self):
        # Connect and start video
        if not self.use_udp:
            try:
                self.tello.connect(wait_for_state=False)
            except TypeError:
                # Older/newer djitellopy versions may not accept the keyword.
                self.tello.connect(False)
            except Exception as e:
                print("\nERROR: Could not connect to the Tello.")
                print("The drone responded to SDK commands, but no state packet was received.")
                raise e

            try:
                print(f"Battery: {self.tello.get_battery()}%")
            except Exception:
                print("Battery: unavailable because no Tello state packet was received.")
            
            self.tello.streamon()

            time.sleep(2)
            frame_reader = self.tello.get_frame_read()
        else:
            frame_reader = None

        print("=" * 50)
        print("  TELLO AUTONOMOUS FOLLOWING")
        print(f"  Mode: {'UDP' if self.use_udp else 'Direct'}")
        print(f"  Detector: {self.detector.mode.upper()}")
        print(f"  YOLO confidence threshold: {self.detector.confidence:.2f}")
        print(f"  Target screen position: x=50%, y={self.target_y_ratio:.0%}")
        print("=" * 50)
        print("  Controls: p=pause  d=switch detector  q=quit")
        print("=" * 50)

        if not self.use_udp:
            print("\nTaking off...")
            self.tello.takeoff()
            time.sleep(2)
            # Move up a bit for better view
            self.tello.move_up(50)
            time.sleep(1)

        print("Following started!\n")

        try:
            while True:
                # Get frame
                if self.use_udp:
                    # When using UDP, we need a separate video source
                    # The tello_bridge.py handles video with --show-video
                    # Here we connect to Tello just for video
                    print("\rWaiting for video... (make sure tello_bridge has --show-video)", end="")
                    time.sleep(0.1)
                    continue
                else:
                    raw_frame = frame_reader.frame
                    if raw_frame is None:
                        time.sleep(0.01)
                        continue

                    # Keep two separate frame paths:
                    #   1. yolo_frame: RGB for YOLO/color detection.
                    #   2. display_frame: BGR for OpenCV HUD drawing/display.
                    #
                    # The test script showed that --source-format rgb displays
                    # correctly, so frame_reader.frame is already RGB in this
                    # environment. Keep the model input as RGB, and convert only
                    # the display copy to BGR because cv2.imshow expects BGR.
                    yolo_frame = raw_frame.copy()
                    display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

                fwd = lat = yaw = vert = 0.0
                fh, fw = yolo_frame.shape[:2]
                fcx = fw // 2
                fcy = int(fh * self.target_y_ratio)

                found, cx, cy, bw, bh, box_xy, label, conf, method = self.detector.detect(yolo_frame)

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
                        print(f"\rTRACKING [{method}] {conf:.0%} | "
                              f"fwd={fwd:+.2f} yaw={yaw:+.2f}    ", end="", flush=True)
                    else:
                        self.lost_count += 1
                        if self.lost_count > self.max_lost:
                            self.send_cmd(yaw=0.2)
                            print(f"\rSEARCHING...    ", end="", flush=True)
                        else:
                            self.stop()

                display = self.draw_hud(display_frame, found, cx, cy, bw, bh,
                                        box_xy, label, conf, method,
                                        fwd, lat, yaw, vert)
                # display_frame is already BGR, which is what cv2.imshow expects.
                cv2.imshow("Tello Auto Follow", display)

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
            print("\n\nStopping...")
            self.stop()
            if not self.use_udp:
                print("Landing...")
                self.tello.land()
                self.tello.streamoff()
            cv2.destroyAllWindows()
            print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Tello autonomous following")
    parser.add_argument("--detector", choices=["yolo", "color", "hybrid"],
                        default="hybrid", help="Detection mode")
    parser.add_argument("--model", default="runs/detect/gazebo_vehicle/weights/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--udp", action="store_true",
                        help="Send commands via UDP to command_conversion.py")
    parser.add_argument("--udp-port", type=int, default=5005,
                        help="UDP port for MotionCommand output")
    parser.add_argument("--confidence", type=float, default=0.7,
                        help="YOLO confidence threshold for accepting robot detections")
    parser.add_argument("--target-y-ratio", type=float, default=0.75,
                        help="Vertical image target as a fraction from top of frame; 0.75 is lower quarter")
    args = parser.parse_args()

    follower = TelloAutoFollow(
        detector_mode=args.detector,
        model_path=args.model,
        use_udp=args.udp,
        udp_port=args.udp_port,
        detection_confidence=args.confidence,
        target_y_ratio=args.target_y_ratio
    )
    follower.run()


if __name__ == "__main__":
    main()
