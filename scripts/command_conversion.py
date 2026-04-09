from dataclasses import dataclass
import argparse
import json
import socket
import threading
import time
from typing import Optional


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class MotionCommand:
    """
    General normalized motion command.

    All values are expected in [-1.0, 1.0]:
      forward > 0 : move forward
      right   > 0 : move right
      up      > 0 : move up
      yaw     > 0 : rotate clockwise/right
    """
    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0

    def clamped(self) -> "MotionCommand":
        return MotionCommand(
            forward=clamp(self.forward, -1.0, 1.0),
            right=clamp(self.right, -1.0, 1.0),
            up=clamp(self.up, -1.0, 1.0),
            yaw=clamp(self.yaw, -1.0, 1.0),
        )

    def as_dict(self) -> dict:
        return {
            "forward": self.forward,
            "right": self.right,
            "up": self.up,
            "yaw": self.yaw,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MotionCommand":
        return cls(
            forward=float(data.get("forward", 0.0)),
            right=float(data.get("right", 0.0)),
            up=float(data.get("up", 0.0)),
            yaw=float(data.get("yaw", 0.0)),
        ).clamped()


@dataclass
class TelloRCCommand:
    """
    Tello RC command format:
      left_right, forward_back, up_down, yaw
    Each value should be in [-100, 100].
    """
    left_right: int = 0
    forward_back: int = 0
    up_down: int = 0
    yaw: int = 0

    def as_tuple(self):
        return (self.left_right, self.forward_back, self.up_down, self.yaw)

    def as_dict(self) -> dict:
        return {
            "left_right": self.left_right,
            "forward_back": self.forward_back,
            "up_down": self.up_down,
            "yaw": self.yaw,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TelloRCCommand":
        return cls(
            left_right=int(data.get("left_right", 0)),
            forward_back=int(data.get("forward_back", 0)),
            up_down=int(data.get("up_down", 0)),
            yaw=int(data.get("yaw", 0)),
        )


class CommandConverter:
    """
    Converts normalized MotionCommand -> TelloRCCommand.

    Features:
    - clamping
    - deadband
    - axis inversion
    - smoothing
    - per-axis gain scaling
    """

    def __init__(
        self,
        max_rc_xy: int = 100,
        max_rc_z: int = 100,
        max_rc_yaw: int = 100,
        deadband: float = 0.05,
        smoothing: float = 0.2,
        invert_forward: bool = False,
        invert_right: bool = False,
        invert_up: bool = False,
        invert_yaw: bool = False,
    ):
        self.max_rc_xy = max_rc_xy
        self.max_rc_z = max_rc_z
        self.max_rc_yaw = max_rc_yaw
        self.deadband = deadband
        self.smoothing = clamp(smoothing, 0.0, 0.95)

        self.invert_forward = invert_forward
        self.invert_right = invert_right
        self.invert_up = invert_up
        self.invert_yaw = invert_yaw

        self._previous = MotionCommand()

    def reset(self):
        self._previous = MotionCommand()

    def _apply_deadband(self, value: float) -> float:
        return 0.0 if abs(value) < self.deadband else value

    def _apply_smoothing(self, current: float, previous: float) -> float:
        alpha = 1.0 - self.smoothing
        return alpha * current + self.smoothing * previous

    def _prepare_motion(self, cmd: MotionCommand) -> MotionCommand:
        cmd = cmd.clamped()

        forward = -cmd.forward if self.invert_forward else cmd.forward
        right = -cmd.right if self.invert_right else cmd.right
        up = -cmd.up if self.invert_up else cmd.up
        yaw = -cmd.yaw if self.invert_yaw else cmd.yaw

        forward = self._apply_deadband(forward)
        right = self._apply_deadband(right)
        up = self._apply_deadband(up)
        yaw = self._apply_deadband(yaw)

        smoothed = MotionCommand(
            forward=self._apply_smoothing(forward, self._previous.forward),
            right=self._apply_smoothing(right, self._previous.right),
            up=self._apply_smoothing(up, self._previous.up),
            yaw=self._apply_smoothing(yaw, self._previous.yaw),
        )

        self._previous = smoothed
        return smoothed

    def convert(self, cmd: MotionCommand) -> TelloRCCommand:
        cmd = self._prepare_motion(cmd)

        left_right = int(round(clamp(cmd.right * self.max_rc_xy, -100, 100)))
        forward_back = int(round(clamp(cmd.forward * self.max_rc_xy, -100, 100)))
        up_down = int(round(clamp(cmd.up * self.max_rc_z, -100, 100)))
        yaw = int(round(clamp(cmd.yaw * self.max_rc_yaw, -100, 100)))

        return TelloRCCommand(
            left_right=left_right,
            forward_back=forward_back,
            up_down=up_down,
            yaw=yaw,
        )


class MotionReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 5005):
        self.cmd = MotionCommand()
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
                cmd = MotionCommand.from_dict(msg)
                with self.lock:
                    self.cmd = cmd
                    self.timestamp = time.time()
            except socket.timeout:
                continue
            except Exception as exc:
                print(f"[WARN] Bad motion packet: {exc}")

    def get_latest(self):
        with self.lock:
            return self.cmd, self.timestamp

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


class RCForwarder:
    def __init__(self, host: str = "127.0.0.1", port: int = 6005):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd: TelloRCCommand):
        payload = json.dumps(cmd.as_dict()).encode("utf-8")
        self.sock.sendto(payload, self.addr)


class ConversionRelay:
    def __init__(
        self,
        input_host: str = "0.0.0.0",
        input_port: int = 5005,
        output_host: str = "127.0.0.1",
        output_port: int = 6005,
        send_hz: float = 20.0,
        timeout: float = 0.5,
        converter: Optional[CommandConverter] = None,
    ):
        self.receiver = MotionReceiver(host=input_host, port=input_port)
        self.forwarder = RCForwarder(host=output_host, port=output_port)
        self.converter = converter or CommandConverter()
        self.send_period = 1.0 / max(send_hz, 1.0)
        self.timeout = max(timeout, 0.0)
        self.running = True

    def run(self):
        self.receiver.start()
        print(
            f"[INFO] Listening for MotionCommand packets on "
            f"{self.receiver.sock.getsockname()[0]}:{self.receiver.sock.getsockname()[1]}"
        )
        print(f"[INFO] Forwarding TelloRCCommand packets to {self.forwarder.addr[0]}:{self.forwarder.addr[1]}")

        try:
            while self.running:
                motion, timestamp = self.receiver.get_latest()
                age = time.time() - timestamp

                if age > self.timeout:
                    self.converter.reset()
                    rc = TelloRCCommand()
                else:
                    rc = self.converter.convert(motion)

                self.forwarder.send(rc)
                print(
                    "\r"
                    f"Motion f={motion.forward:+.2f} r={motion.right:+.2f} "
                    f"u={motion.up:+.2f} y={motion.yaw:+.2f}  ->  "
                    f"RC lr={rc.left_right:+4d} fb={rc.forward_back:+4d} "
                    f"ud={rc.up_down:+4d} yw={rc.yaw:+4d}",
                    end="",
                    flush=True,
                )
                time.sleep(self.send_period)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping conversion relay.")
        finally:
            try:
                self.forwarder.send(TelloRCCommand())
            except OSError:
                pass
            self.receiver.stop()
            print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(description="Convert MotionCommand UDP packets into Tello RC UDP packets")
    parser.add_argument("--input-host", default="0.0.0.0", help="Host/interface to listen on for motion commands")
    parser.add_argument("--input-port", type=int, default=5005, help="UDP port for MotionCommand input")
    parser.add_argument("--output-host", default="127.0.0.1", help="Host running tello_bridge")
    parser.add_argument("--output-port", type=int, default=6005, help="UDP port tello_bridge listens on for RC commands")
    parser.add_argument("--send-hz", type=float, default=20.0, help="How often to forward RC commands")
    parser.add_argument("--timeout", type=float, default=0.5, help="Seconds before stale input is zeroed")
    parser.add_argument("--max-rc-xy", type=int, default=100)
    parser.add_argument("--max-rc-z", type=int, default=100)
    parser.add_argument("--max-rc-yaw", type=int, default=100)
    parser.add_argument("--deadband", type=float, default=0.05)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--invert-forward", action="store_true")
    parser.add_argument("--invert-right", action="store_true")
    parser.add_argument("--invert-up", action="store_true")
    parser.add_argument("--invert-yaw", action="store_true")
    args = parser.parse_args()

    relay = ConversionRelay(
        input_host=args.input_host,
        input_port=args.input_port,
        output_host=args.output_host,
        output_port=args.output_port,
        send_hz=args.send_hz,
        timeout=args.timeout,
        converter=CommandConverter(
            max_rc_xy=args.max_rc_xy,
            max_rc_z=args.max_rc_z,
            max_rc_yaw=args.max_rc_yaw,
            deadband=args.deadband,
            smoothing=args.smoothing,
            invert_forward=args.invert_forward,
            invert_right=args.invert_right,
            invert_up=args.invert_up,
            invert_yaw=args.invert_yaw,
        ),
    )
    relay.run()


if __name__ == "__main__":
    main()
