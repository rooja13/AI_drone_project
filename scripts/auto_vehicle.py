#!/usr/bin/env python3
"""
Auto-drive the ground vehicle in configurable patterns.
This gives the drone a target to track and follow.

Usage:
    python3 scripts/auto_vehicle.py                     # default circle
    python3 scripts/auto_vehicle.py --pattern square
    python3 scripts/auto_vehicle.py --pattern straight
    python3 scripts/auto_vehicle.py --pattern random
    python3 scripts/auto_vehicle.py --pattern circle --speed 1.5

Patterns:
    circle   - continuous circular driving
    square   - drives in a square path with stops at corners
    straight - drives forward and reverses
    random   - random turns and speed changes (more realistic)
"""

import time
import random
import argparse
from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist


class VehicleDriver:
    def __init__(self, speed=1.0):
        self.node = Node()
        self.pub = self.node.advertise("/ground_vehicle/cmd_vel", Twist)
        self.speed = speed
        time.sleep(0.5)  # let Gazebo register the publisher

    def send(self, linear_x=0.0, angular_z=0.0):
        """Send velocity command to the ground vehicle."""
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.pub.publish(twist)

    def stop(self):
        """Stop the vehicle."""
        self.send(0.0, 0.0)

    def drive_circle(self):
        """Drive in a continuous circle."""
        print("Pattern: CIRCLE")
        print(f"  Speed: {self.speed} m/s | Turn rate: {self.speed * 0.5} rad/s")
        while True:
            self.send(self.speed, self.speed * 0.5)
            time.sleep(0.05)

    def drive_square(self):
        """Drive in a square pattern with pauses at corners."""
        print("Pattern: SQUARE")
        side_duration = 3.0
        turn_duration = 1.57

        while True:
            print("  → Driving forward...")
            t = 0
            while t < side_duration:
                self.send(self.speed, 0.0)
                time.sleep(0.05)
                t += 0.05

            self.stop()
            time.sleep(0.3)

            print("  ↻ Turning...")
            t = 0
            while t < turn_duration:
                self.send(0.0, 1.0)
                time.sleep(0.05)
                t += 0.05

            self.stop()
            time.sleep(0.3)

    def drive_straight(self):
        """Drive forward and backward in a line."""
        print("Pattern: STRAIGHT")
        drive_duration = 4.0

        while True:
            print("  → Forward")
            t = 0
            while t < drive_duration:
                self.send(self.speed, 0.0)
                time.sleep(0.05)
                t += 0.05

            self.stop()
            time.sleep(0.5)

            print("  ← Reverse")
            t = 0
            while t < drive_duration:
                self.send(-self.speed, 0.0)
                time.sleep(0.05)
                t += 0.05

            self.stop()
            time.sleep(0.5)

    def drive_random(self):
        """Drive with random speed and direction changes (more realistic)."""
        print("Pattern: RANDOM")
        print("  Vehicle will change speed and direction randomly")

        while True:
            fwd = random.uniform(0.3, 1.0) * self.speed
            turn = random.uniform(-1.0, 1.0)
            duration = random.uniform(1.0, 4.0)

            direction = "straight" if abs(turn) < 0.3 else ("left" if turn > 0 else "right")
            print(f"  → speed={fwd:.1f} turn={turn:.1f} ({direction}) for {duration:.1f}s")

            t = 0
            while t < duration:
                self.send(fwd, turn)
                time.sleep(0.05)
                t += 0.05

            if random.random() < 0.2:
                print("  ■ Pausing...")
                self.stop()
                time.sleep(random.uniform(0.5, 1.5))


def main():
    parser = argparse.ArgumentParser(
        description="Auto-drive the ground vehicle for drone tracking"
    )
    parser.add_argument(
        "--pattern",
        choices=["circle", "square", "straight", "random"],
        default="circle",
        help="Driving pattern (default: circle)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Base forward speed in m/s (default: 1.0)"
    )
    args = parser.parse_args()

    driver = VehicleDriver(speed=args.speed)

    print(f"Ground Vehicle Auto-Driver")
    print(f"Speed: {args.speed} m/s")
    print(f"Make sure Gazebo simulation is playing!")
    print(f"Press Ctrl+C to stop.\n")

    try:
        if args.pattern == "circle":
            driver.drive_circle()
        elif args.pattern == "square":
            driver.drive_square()
        elif args.pattern == "straight":
            driver.drive_straight()
        elif args.pattern == "random":
            driver.drive_random()
    except KeyboardInterrupt:
        driver.stop()
        print("\n\nVehicle stopped.")


if __name__ == "__main__":
    main()
