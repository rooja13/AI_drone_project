import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass

import cv2
from djitellopy import Tello


@dataclass
class SimCommand:
    vx: float = 0.0       # forward/back, m/s-ish
    vy: float = 0.0       # left/right, m/s-ish
    vz: float = 0.0       # up/down, m/s-ish
    yaw_rate: float = 0.0 # rad/s-ish
    timestamp: float = 0.0


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class SimReceiver:
    def __init__(self, host="0.0.0.0", port=5005):
        self.cmd = SimCommand(timestamp=time.time())
        self.lock = threading.Lock()
        self.running = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        return t

    def _run(self):
        while self.running:
            try:
                data, _addr = self.sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                new_cmd = SimCommand(
                    vx=float(msg.get("vx", 0.0)),
                    vy=float(msg.get("vy", 0.0)),
                    vz=float(msg.get("vz", 0.0)),
                    yaw_rate=float(msg.get("yaw_rate", 0.0)),
                    timestamp=time.time(),
                )
                with self.lock:
                    self.cmd = new_cmd
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[WARN] Bad sim packet: {e}")

    def get_latest(self):
        with self.lock:
            return SimCommand(
                vx=self.cmd.vx,
                vy=self.cmd.vy,
                vz=self.cmd.vz,
                yaw_rate=self.cmd.yaw_rate,
                timestamp=self.cmd.timestamp,
            )

    def stop(self):
        self.running = False
        self.sock.close()


def video_loop(tello, stop_event):
    frame_reader = tello.get_frame_read()

    while not stop_event.is_set():
        frame = frame_reader.frame
        if frame is None:
            time.sleep(0.01)
            continue

        display = frame.copy()
        cv2.putText(
            display,
            "Press q in video window to stop display",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Tello Live Feed", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-port", type=int, default=5005)
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--takeoff", action="store_true")
    parser.add_argument("--k-xy", type=float, default=80.0)
    parser.add_argument("--k-z", type=float, default=80.0)
    parser.add_argument("--k-yaw", type=float, default=50.0)
    parser.add_argument("--timeout", type=float, default=0.5)
    args = parser.parse_args()

    tello = Tello()
    print("[INFO] Connecting to Tello...")
    tello.connect()
    print(f"[INFO] Battery: {tello.get_battery()}%")

    print("[INFO] Starting video stream...")
    tello.streamon()

    stop_event = threading.Event()

    if args.show_video:
        vt = threading.Thread(target=video_loop, args=(tello, stop_event), daemon=True)
        vt.start()

    sim = SimReceiver(port=args.sim_port)
    sim.start()
    print(f"[INFO] Listening for sim commands on UDP port {args.sim_port}")

    try:
        if args.takeoff:
            print("[INFO] Taking off...")
            tello.takeoff()
            time.sleep(2)

        print("[INFO] Sending RC control. Ctrl+C to stop.")
        while not stop_event.is_set():
            cmd = sim.get_latest()
            age = time.time() - cmd.timestamp

            if age > args.timeout:
                lr = fb = ud = yw = 0
            else:
                lr = int(clamp(args.k_xy * cmd.vy, -100, 100))
                fb = int(clamp(args.k_xy * cmd.vx, -100, 100))
                ud = int(clamp(args.k_z * cmd.vz, -100, 100))
                yw = int(clamp(args.k_yaw * cmd.yaw_rate, -100, 100))

            tello.send_rc_control(lr, fb, ud, yw)
            print(f"\rRC => lr={lr:4d} fb={fb:4d} ud={ud:4d} yw={yw:4d}", end="")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("\n[INFO] Stopping drone motion...")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.2)
        except Exception:
            pass

        try:
            if tello.is_flying:
                print("[INFO] Landing...")
                tello.land()
        except Exception:
            pass

        sim.stop()

        try:
            tello.streamoff()
        except Exception:
            pass

        print("[INFO] Done.")


if __name__ == "__main__":
    main()
