import argparse
import json
import socket
import threading
import time
from typing import Set

from pynput import keyboard

from command_conversion import MotionCommand, clamp


class KeyboardMotionSender:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5005,
        linear_speed: float = 0.6,
        vertical_speed: float = 0.6,
        yaw_speed: float = 0.6,
        send_hz: float = 20.0,
    ):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.linear_speed = clamp(linear_speed, 0.0, 1.0)
        self.vertical_speed = clamp(vertical_speed, 0.0, 1.0)
        self.yaw_speed = clamp(yaw_speed, 0.0, 1.0)
        self.send_period = 1.0 / max(send_hz, 1.0)

        self.pressed: Set[object] = set()
        self.lock = threading.Lock()
        self.running = True

    def _is_pressed(self, key) -> bool:
        return key in self.pressed

    def _current_command(self) -> MotionCommand:
        with self.lock:
            forward = 0.0
            right = 0.0
            up = 0.0
            yaw = 0.0

            if self._is_pressed("w"):
                forward += self.linear_speed
            if self._is_pressed("s"):
                forward -= self.linear_speed

            if self._is_pressed("d"):
                right += self.linear_speed
            if self._is_pressed("a"):
                right -= self.linear_speed

            if self._is_pressed("q"):
                up += self.vertical_speed
            if self._is_pressed("e"):
                up -= self.vertical_speed

            if self._is_pressed(keyboard.Key.right):
                yaw += self.yaw_speed
            if self._is_pressed(keyboard.Key.left):
                yaw -= self.yaw_speed

            return MotionCommand(
                forward=clamp(forward, -1.0, 1.0),
                right=clamp(right, -1.0, 1.0),
                up=clamp(up, -1.0, 1.0),
                yaw=clamp(yaw, -1.0, 1.0),
            )

    def _send_loop(self):
        while self.running:
            cmd = self._current_command()
            payload = json.dumps(cmd.as_dict()).encode("utf-8")
            self.sock.sendto(payload, self.addr)
            print(
                "\r"
                f"Sending MotionCommand to converter at {self.addr[0]}:{self.addr[1]} | "
                f"forward={cmd.forward:+.2f} right={cmd.right:+.2f} "
                f"up={cmd.up:+.2f} yaw={cmd.yaw:+.2f}",
                end="",
                flush=True,
            )
            time.sleep(self.send_period)

    def _normalize_key(self, key):
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        return key

    def on_press(self, key):
        key = self._normalize_key(key)
        with self.lock:
            self.pressed.add(key)

        if key == keyboard.Key.esc:
            self.running = False
            return False

        if key == " ":
            with self.lock:
                self.pressed.clear()

    def on_release(self, key):
        key = self._normalize_key(key)
        with self.lock:
            self.pressed.discard(key)

    def send_zero_once(self):
        zero = MotionCommand()
        self.sock.sendto(json.dumps(zero.as_dict()).encode("utf-8"), self.addr)

    def run(self):
        sender_thread = threading.Thread(target=self._send_loop, daemon=True)
        sender_thread.start()

        print("Controls:")
        print("  W/S : forward/back")
        print("  A/D : left/right")
        print("  Q/E : up/down")
        print("  Left/Right arrows : yaw")
        print("  Space : stop motion")
        print("  Esc : quit")
        print(f"Sending MotionCommand UDP packets to converter at {self.addr[0]}:{self.addr[1]}")

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        listener.join()

        self.running = False
        sender_thread.join(timeout=1.0)
        self.send_zero_once()
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Keyboard WASD controller that sends MotionCommand packets to command_conversion.py")
    parser.add_argument("--host", default="127.0.0.1", help="Host running command_conversion.py")
    parser.add_argument("--port", type=int, default=5005, help="UDP port used by command_conversion.py for MotionCommand input")
    parser.add_argument("--linear-speed", type=float, default=0.6, help="Normalized WASD speed in [0,1]")
    parser.add_argument("--vertical-speed", type=float, default=0.6, help="Normalized Q/E speed in [0,1]")
    parser.add_argument("--yaw-speed", type=float, default=0.5, help="Normalized yaw speed in [0,1]")
    parser.add_argument("--send-hz", type=float, default=20.0, help="How often to resend commands")
    args = parser.parse_args()

    sender = KeyboardMotionSender(
        host=args.host,
        port=args.port,
        linear_speed=args.linear_speed,
        vertical_speed=args.vertical_speed,
        yaw_speed=args.yaw_speed,
        send_hz=args.send_hz,
    )
    sender.run()


if __name__ == "__main__":
    main()
