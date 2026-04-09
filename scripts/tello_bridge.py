import argparse
import json
import socket
import threading
import time

import cv2
from djitellopy import Tello

from command_conversion import TelloRCCommand


class RCReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 6005):
        self.cmd = TelloRCCommand()
        self.timestamp = time.time()
        self.lock = threading.Lock()
        self.running = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return thread

    def _run(self):
        while self.running:
            try:
                data, _addr = self.sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                cmd = TelloRCCommand.from_dict(msg)
                with self.lock:
                    self.cmd = cmd
                    self.timestamp = time.time()
            except socket.timeout:
                continue
            except Exception as exc:
                print(f"[WARN] Bad RC packet: {exc}")

    def get_latest(self):
        with self.lock:
            return self.cmd, self.timestamp

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


def video_loop(tello: Tello, stop_event: threading.Event):
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
    parser = argparse.ArgumentParser(
        description="Receives TelloRCCommand packets and sends them to a real Tello drone"
    )
    parser.add_argument("--rc-host", default="0.0.0.0", help="Host/interface to bind for RC packet input")
    parser.add_argument("--rc-port", type=int, default=6005, help="UDP port for TelloRCCommand input")
    parser.add_argument("--show-video", action="store_true", help="Display live Tello camera feed")
    parser.add_argument("--takeoff", action="store_true", help="Take off automatically after connecting")
    parser.add_argument("--timeout", type=float, default=0.5, help="Seconds before stale RC commands are zeroed")
    parser.add_argument("--send-hz", type=float, default=10.0, help="How often to send RC commands to the drone")
    args = parser.parse_args()

    send_period = 1.0 / max(args.send_hz, 1.0)

    tello = Tello()
    print("[INFO] Connecting to Tello...")
    tello.connect()
    print(f"[INFO] Battery: {tello.get_battery()}%")

    print("[INFO] Starting video stream...")
    tello.streamon()

    stop_event = threading.Event()

    if args.show_video:
        video_thread = threading.Thread(target=video_loop, args=(tello, stop_event), daemon=True)
        video_thread.start()

    receiver = RCReceiver(host=args.rc_host, port=args.rc_port)
    receiver.start()
    print(f"[INFO] Listening for TelloRCCommand packets on UDP {args.rc_host}:{args.rc_port}")

    try:
        if args.takeoff:
            print("[INFO] Taking off...")
            tello.takeoff()
            time.sleep(2)

        print("[INFO] Sending RC control. Ctrl+C to stop.")
        while not stop_event.is_set():
            cmd, timestamp = receiver.get_latest()
            age = time.time() - timestamp

            if age > args.timeout:
                lr = fb = ud = yw = 0
            else:
                lr, fb, ud, yw = cmd.as_tuple()

            tello.send_rc_control(lr, fb, ud, yw)
            print(
                f"\rRC => lr={lr:4d} fb={fb:4d} ud={ud:4d} yw={yw:4d} age={age:0.2f}s",
                end="",
                flush=True,
            )
            time.sleep(send_period)

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

        receiver.stop()

        try:
            tello.streamoff()
        except Exception:
            pass

        print("[INFO] Done.")


if __name__ == "__main__":
    main()
