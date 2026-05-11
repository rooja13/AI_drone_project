#!/usr/bin/env python3
"""
Drive the three Gazebo ground vehicles in bounded patrol patterns with
Ackermann/car-safe steering and detailed debugging.

Why this version exists:
  The Gazebo vehicles do not reliably rotate when commanded with
  linear.x = 0 and angular.z != 0. They behave like car/ackermann-style
  vehicles, so they need forward or reverse motion while steering.

Default vehicles/topics:
  RED   model: vehicle_red    topic: /ground_vehicle/cmd_vel
  BLUE  model: vehicle_blue   topic: /ground_vehicle_blue/cmd_vel
  GREEN model: vehicle_green  topic: /ground_vehicle_green/cmd_vel

Default movement:
  RED   = bounded clockwise circle around its spawn area
  BLUE  = bounded counter-clockwise circle around its spawn area
  GREEN = bounded figure-eight around its spawn area

Usage:
  python3 scripts/auto_vehicle_multi.py
  python3 scripts/auto_vehicle_multi.py --debug-every 0.5
  python3 scripts/auto_vehicle_multi.py --csv-log vehicle_debug.csv

Useful tuning:
  python3 scripts/auto_vehicle_multi.py --max-turn 0.9 --steer-kp 1.4
  python3 scripts/auto_vehicle_multi.py --lookahead-steps 3 --min-turn-speed 0.35
"""

import argparse
import csv
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist

try:
    from gz.msgs10.pose_v_pb2 import Pose_V
except Exception:
    Pose_V = None


DT = 0.05


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class PoseSample:
    x: float
    y: float
    yaw: float
    timestamp: float
    speed: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class DriverState:
    name: str
    model_name: str
    topic: str
    pattern: str
    phase: str = "starting"
    cmd_linear_x: float = 0.0
    cmd_angular_z: float = 0.0
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    waypoint_index: int = 0
    nearest_index: int = 0
    distance_to_target: Optional[float] = None
    heading_error: Optional[float] = None
    area_error: Optional[float] = None
    turn_response: str = "--"
    command_count: int = 0
    last_send_time: float = field(default_factory=time.monotonic)


class PoseMonitor:
    """Subscribes to Gazebo world pose info and stores model poses."""

    def __init__(self, pose_topic: str, model_names: Iterable[str]):
        self.pose_topic = pose_topic
        self.model_names = set(model_names)
        self.poses: Dict[str, PoseSample] = {}
        self._lock = threading.Lock()
        self.subscribed = False
        self.node = None

        if Pose_V is None:
            print("WARNING: Could not import Pose_V; pose debug disabled.")
            return

        self.node = Node()
        try:
            self.subscribed = bool(self.node.subscribe(Pose_V, pose_topic, self._pose_callback))
        except Exception as exc:
            print(f"WARNING: Could not subscribe to {pose_topic}: {exc}")
            self.subscribed = False

        if self.subscribed:
            print(f"Pose debug: subscribed to {pose_topic}")
        else:
            print(f"WARNING: Pose debug subscription failed for {pose_topic}")

    def _pose_callback(self, msg):
        now = time.monotonic()
        with self._lock:
            for pose in msg.pose:
                if pose.name not in self.model_names:
                    continue

                x = float(pose.position.x)
                y = float(pose.position.y)
                yaw = yaw_from_quaternion(pose.orientation)

                prev = self.poses.get(pose.name)
                speed = 0.0
                yaw_rate = 0.0
                if prev is not None:
                    dt = max(now - prev.timestamp, 1e-6)
                    speed = math.hypot(x - prev.x, y - prev.y) / dt
                    yaw_rate = wrap_pi(yaw - prev.yaw) / dt

                self.poses[pose.name] = PoseSample(
                    x=x,
                    y=y,
                    yaw=yaw,
                    timestamp=now,
                    speed=speed,
                    yaw_rate=yaw_rate,
                )

    def get(self, model_name: str) -> Optional[PoseSample]:
        with self._lock:
            return self.poses.get(model_name)


def circle_waypoints(center: Tuple[float, float], radius: float, count: int, clockwise: bool) -> List[Tuple[float, float]]:
    cx, cy = center
    sign = -1.0 if clockwise else 1.0
    points = []
    for i in range(count):
        theta = sign * 2.0 * math.pi * i / count
        points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return points


def figure_eight_waypoints(center: Tuple[float, float], radius: float, count: int) -> List[Tuple[float, float]]:
    cx, cy = center
    points = []
    for i in range(count):
        t = 2.0 * math.pi * i / count
        # Gerono lemniscate: compact, bounded, crosses at the center.
        x = cx + radius * math.sin(t)
        y = cy + radius * math.sin(t) * math.cos(t)
        points.append((x, y))
    return points


def nearest_waypoint_index(points: Sequence[Tuple[float, float]], x: float, y: float) -> int:
    best_i = 0
    best_d = float("inf")
    for i, (px, py) in enumerate(points):
        d = math.hypot(px - x, py - y)
        if d < best_d:
            best_i = i
            best_d = d
    return best_i


class VehicleDriver:
    """Pure-pursuit style rolling-turn controller for one Gazebo vehicle."""

    def __init__(
        self,
        state: DriverState,
        speed: float,
        waypoints: Sequence[Tuple[float, float]],
        area_center: Tuple[float, float],
        area_radius: float,
        dry_run: bool = False,
    ):
        self.state = state
        self.speed = float(speed)
        self.waypoints = list(waypoints)
        self.area_center = area_center
        self.area_radius = float(area_radius)
        self.dry_run = dry_run
        self.running = True
        self.node = Node()
        self.pub = None
        self._initialized = False
        self._last_good_yaw_time = time.monotonic()
        self._recovery_until = 0.0
        self._recovery_sign = 1.0

        if not dry_run:
            self.pub = self.node.advertise(state.topic, Twist)
            time.sleep(0.5)

    def send(self, linear_x: float, angular_z: float):
        linear_x = float(linear_x)
        angular_z = float(angular_z)
        self.state.cmd_linear_x = linear_x
        self.state.cmd_angular_z = angular_z
        self.state.command_count += 1
        self.state.last_send_time = time.monotonic()

        if self.dry_run:
            return

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.pub.publish(twist)

    def stop(self):
        self.send(0.0, 0.0)

    def _select_target(
        self,
        pose: PoseSample,
        lookahead_steps: int,
        max_area_error: float,
    ) -> Tuple[float, float, str]:
        center_x, center_y = self.area_center
        dist_from_center = math.hypot(pose.x - center_x, pose.y - center_y)
        self.state.area_error = dist_from_center - self.area_radius

        if dist_from_center > self.area_radius + max_area_error:
            # If it escaped the patrol area, do not chase the orbit. Roll back
            # toward the center while steering; never turn in place.
            return center_x, center_y, "return_to_area"

        nearest = nearest_waypoint_index(self.waypoints, pose.x, pose.y)
        self.state.nearest_index = nearest
        self.state.waypoint_index = (nearest + lookahead_steps) % len(self.waypoints)
        return (*self.waypoints[self.state.waypoint_index], "rolling_patrol")

    def drive(
        self,
        pose_monitor: PoseMonitor,
        open_loop: bool,
        min_turn_speed: float,
        max_turn: float,
        steer_kp: float,
        lookahead_steps: int,
        max_area_error: float,
        slow_for_turn: float,
        enable_unstick: bool,
        unstick_after: float,
        unstick_seconds: float,
    ):
        open_loop_sign = -1.0 if "cw" in self.state.pattern else 1.0
        if "figure" in self.state.pattern:
            open_loop_sign = 1.0
        open_loop_start = time.monotonic()

        while self.running:
            pose = None if open_loop else pose_monitor.get(self.state.model_name)

            if pose is None:
                # No pose feedback: still use car-safe rolling motion.
                elapsed = time.monotonic() - open_loop_start
                if "figure" in self.state.pattern:
                    sign = 1.0 if int(elapsed // 4.0) % 2 == 0 else -1.0
                    turn = sign * max_turn
                else:
                    turn = open_loop_sign * max_turn
                self.state.phase = "open_loop_roll"
                self.state.target_x = None
                self.state.target_y = None
                self.state.distance_to_target = None
                self.state.heading_error = None
                self.state.area_error = None
                self.state.turn_response = "no_pose"
                self.send(self.speed, turn)
                time.sleep(DT)
                continue

            target_x, target_y, phase = self._select_target(pose, lookahead_steps, max_area_error)
            dx = target_x - pose.x
            dy = target_y - pose.y
            distance = math.hypot(dx, dy)
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_pi(desired_heading - pose.yaw)

            self.state.target_x = target_x
            self.state.target_y = target_y
            self.state.distance_to_target = distance
            self.state.heading_error = heading_error

            now = time.monotonic()
            turn = clamp(steer_kp * heading_error, -max_turn, max_turn)

            # Ackermann/car-safe rule: do not use pure in-place turns.
            # Slow down during large heading corrections, but keep moving.
            turn_fraction = min(abs(heading_error) / math.pi, 1.0)
            linear = self.speed * (1.0 - slow_for_turn * turn_fraction)
            linear = max(min_turn_speed, linear)

            if phase == "return_to_area":
                linear = min(linear, 0.75 * self.speed)

            if enable_unstick:
                # If steering is commanded while rolling but yaw stays near zero,
                # briefly reverse while steering. This still respects the car-safe
                # rule because linear.x is nonzero.
                turning_commanded = abs(turn) > 0.25 and abs(linear) > 0.05
                yaw_responding = abs(pose.yaw_rate) > 0.035
                if turning_commanded and yaw_responding:
                    self._last_good_yaw_time = now
                    self.state.turn_response = "OK"
                elif turning_commanded:
                    self.state.turn_response = "LOW"
                    if now - self._last_good_yaw_time > unstick_after and now >= self._recovery_until:
                        self._recovery_until = now + unstick_seconds
                        self._recovery_sign = -1.0 if turn > 0.0 else 1.0
                        self._last_good_yaw_time = now
                else:
                    self.state.turn_response = "--"

                if now < self._recovery_until:
                    self.state.phase = "reverse_roll_unstick"
                    self.send(-0.45 * self.speed, self._recovery_sign * max_turn)
                    time.sleep(DT)
                    continue

            self.state.phase = phase
            self.send(linear, turn)
            time.sleep(DT)


class DebugPrinter:
    def __init__(self, states: List[DriverState], pose_monitor: PoseMonitor, every: float, csv_path: Optional[str]):
        self.states = states
        self.pose_monitor = pose_monitor
        self.every = float(every)
        self.csv_path = csv_path
        self.running = True
        self._csv_file = None
        self._csv_writer = None

        if csv_path:
            directory = os.path.dirname(os.path.abspath(csv_path))
            os.makedirs(directory, exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "time_monotonic", "name", "model_name", "topic", "pattern", "phase",
                "cmd_linear_x", "cmd_angular_z", "pose_x", "pose_y", "pose_yaw_rad",
                "measured_speed", "measured_yaw_rate", "target_x", "target_y",
                "nearest_index", "waypoint_index", "distance_to_target", "heading_error_rad",
                "area_error_m", "turn_response", "command_count",
            ])

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    @staticmethod
    def fmt(value: Optional[float], width: int = 6, precision: int = 2) -> str:
        if value is None:
            return " " * (width - 2) + "--"
        return f"{value:{width}.{precision}f}"

    def print_once(self):
        now = time.monotonic()
        print("\n" + "=" * 142)
        print("Robot command inputs and measured Gazebo motion")
        print("This version never uses linear.x=0 for steering. It uses rolling turns only.")
        print("turn=LOW means cmd_w is nonzero but measured yaw_rate is very small while the robot is moving.")
        print("name  phase                 cmd_v  cmd_w | pose_x pose_y  yaw yaw_rt meas_v | tgt_x tgt_y near wp dist hdg_err area_err turn | topic")
        print("-" * 142)

        for state in self.states:
            pose = self.pose_monitor.get(state.model_name)
            if pose is None:
                px = py = yaw = speed = yaw_rate = None
            else:
                px = pose.x
                py = pose.y
                yaw = pose.yaw
                speed = pose.speed
                yaw_rate = pose.yaw_rate

            print(
                f"{state.name:<5} {state.phase:<21} "
                f"{state.cmd_linear_x:5.2f} {state.cmd_angular_z:6.2f} | "
                f"{self.fmt(px)} {self.fmt(py)} {self.fmt(yaw, 5, 2)} {self.fmt(yaw_rate, 6, 2)} {self.fmt(speed)} | "
                f"{self.fmt(state.target_x)} {self.fmt(state.target_y)} "
                f"{state.nearest_index:4d} {state.waypoint_index:2d} {self.fmt(state.distance_to_target)} "
                f"{self.fmt(state.heading_error)} {self.fmt(state.area_error)} {state.turn_response:>4} | "
                f"{state.topic}"
            )

            if self._csv_writer:
                self._csv_writer.writerow([
                    now, state.name, state.model_name, state.topic, state.pattern, state.phase,
                    state.cmd_linear_x, state.cmd_angular_z, px, py, yaw, speed, yaw_rate,
                    state.target_x, state.target_y, state.nearest_index, state.waypoint_index,
                    state.distance_to_target, state.heading_error, state.area_error,
                    state.turn_response, state.command_count,
                ])
                self._csv_file.flush()

        print("=" * 142)

    def run(self):
        if self.every <= 0.0:
            return
        while self.running:
            self.print_once()
            time.sleep(self.every)


def build_vehicle_specs(args):
    red_center = (args.red_center_x, args.red_center_y)
    blue_center = (args.blue_center_x, args.blue_center_y)
    green_center = (args.green_center_x, args.green_center_y)

    red_points = circle_waypoints(red_center, args.red_radius, args.circle_points, clockwise=True)
    blue_points = circle_waypoints(blue_center, args.blue_radius, args.circle_points, clockwise=False)
    green_points = figure_eight_waypoints(green_center, args.green_radius, args.figure8_points)

    return [
        {
            "state": DriverState("RED", "vehicle_red", args.red_topic, "circle_cw"),
            "speed": args.red_speed,
            "waypoints": red_points,
            "center": red_center,
            "area_radius": args.red_radius,
        },
        {
            "state": DriverState("BLUE", "vehicle_blue", args.blue_topic, "circle_ccw"),
            "speed": args.blue_speed,
            "waypoints": blue_points,
            "center": blue_center,
            "area_radius": args.blue_radius,
        },
        {
            "state": DriverState("GREEN", "vehicle_green", args.green_topic, "figure_eight"),
            "speed": args.green_speed,
            "waypoints": green_points,
            "center": green_center,
            "area_radius": args.green_radius,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Ackermann-safe bounded three-vehicle Gazebo driver with debug output.")

    parser.add_argument("--red-topic", default="/ground_vehicle/cmd_vel")
    parser.add_argument("--blue-topic", default="/ground_vehicle_blue/cmd_vel")
    parser.add_argument("--green-topic", default="/ground_vehicle_green/cmd_vel")

    parser.add_argument("--red-speed", type=float, default=0.70)
    parser.add_argument("--blue-speed", type=float, default=0.65)
    parser.add_argument("--green-speed", type=float, default=0.60)
    parser.add_argument("--min-turn-speed", type=float, default=0.30, help="Minimum forward speed used while steering.")

    parser.add_argument("--red-center-x", type=float, default=5.0)
    parser.add_argument("--red-center-y", type=float, default=0.0)
    parser.add_argument("--blue-center-x", type=float, default=-5.0)
    parser.add_argument("--blue-center-y", type=float, default=3.0)
    parser.add_argument("--green-center-x", type=float, default=3.0)
    parser.add_argument("--green-center-y", type=float, default=-5.0)

    parser.add_argument("--red-radius", type=float, default=1.8)
    parser.add_argument("--blue-radius", type=float, default=1.8)
    parser.add_argument("--green-radius", type=float, default=1.7)
    parser.add_argument("--circle-points", type=int, default=32)
    parser.add_argument("--figure8-points", type=int, default=48)

    parser.add_argument("--lookahead-steps", type=int, default=4)
    parser.add_argument("--max-turn", type=float, default=0.75, help="Max Twist.angular.z command while rolling.")
    parser.add_argument("--steer-kp", type=float, default=1.15)
    parser.add_argument("--slow-for-turn", type=float, default=0.45, help="How much to reduce speed during large heading error, 0 to 1.")
    parser.add_argument("--max-area-error", type=float, default=1.2)

    parser.add_argument("--pose-topic", default="/world/drone_tracker/pose/info")
    parser.add_argument("--open-loop", action="store_true", help="Ignore pose feedback and drive fixed loops/figure-eight.")
    parser.add_argument("--debug-every", type=float, default=0.75)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--disable-unstick", action="store_true")
    parser.add_argument("--unstick-after", type=float, default=2.0)
    parser.add_argument("--unstick-seconds", type=float, default=0.6)

    args = parser.parse_args()

    vehicle_specs = build_vehicle_specs(args)
    states = [spec["state"] for spec in vehicle_specs]
    model_names = [state.model_name for state in states]

    pose_monitor = PoseMonitor(args.pose_topic, model_names)
    use_open_loop = args.open_loop or not pose_monitor.subscribed
    printer = DebugPrinter(states, pose_monitor, args.debug_every, args.csv_log)

    drivers: List[VehicleDriver] = []
    threads: List[threading.Thread] = []

    print("=" * 92)
    print("  GAZEBO MULTI-VEHICLE ACKERMANN-SAFE DEBUG DRIVER")
    print("=" * 92)
    print("  Movement is bounded: red/blue circles, green figure-eight.")
    print("  Controller uses rolling turns only: linear.x stays nonzero while angular.z steers.")
    print("  This matches the log evidence that in-place turns do not rotate these vehicles.")
    print("  Watch turn column: LOW means Gazebo is still not yawing much for the command.")
    if args.csv_log:
        print(f"  CSV debug log: {args.csv_log}")
    if use_open_loop:
        print("  WARNING: pose feedback disabled/unavailable; bounded patrol cannot be guaranteed.")
    print("=" * 92)

    try:
        for spec in vehicle_specs:
            state = spec["state"]
            driver = VehicleDriver(
                state=state,
                speed=spec["speed"],
                waypoints=spec["waypoints"],
                area_center=spec["center"],
                area_radius=spec["area_radius"],
                dry_run=args.dry_run,
            )
            drivers.append(driver)

            thread = threading.Thread(
                target=driver.drive,
                kwargs={
                    "pose_monitor": pose_monitor,
                    "open_loop": use_open_loop,
                    "min_turn_speed": abs(args.min_turn_speed),
                    "max_turn": abs(args.max_turn),
                    "steer_kp": args.steer_kp,
                    "lookahead_steps": max(1, args.lookahead_steps),
                    "max_area_error": abs(args.max_area_error),
                    "slow_for_turn": clamp(args.slow_for_turn, 0.0, 0.95),
                    "enable_unstick": not args.disable_unstick,
                    "unstick_after": args.unstick_after,
                    "unstick_seconds": args.unstick_seconds,
                },
                daemon=True,
            )
            threads.append(thread)

            print(
                f"  {state.name:<5} model={state.model_name:<14} topic={state.topic:<30} "
                f"pattern={state.pattern:<12} speed={spec['speed']:.2f} "
                f"radius={spec['area_radius']:.2f} waypoints={len(spec['waypoints'])}"
            )

        print("=" * 92)
        print("  Ctrl+C stops all vehicles cleanly.")
        print("=" * 92)

        printer_thread = threading.Thread(target=printer.run, daemon=True)
        printer_thread.start()

        for thread in threads:
            thread.start()

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping all vehicles...")
        for driver in drivers:
            driver.running = False
            driver.stop()
        time.sleep(0.5)
        print("Done!")
    finally:
        printer.running = False
        printer.close()


if __name__ == "__main__":
    main()
