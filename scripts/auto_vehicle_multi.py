#!/usr/bin/env python3
"""
Drive multiple ground vehicles simultaneously in different patterns.

Usage:
    python3 scripts/auto_vehicle_multi.py
"""

import time
import random
import threading
from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist


class VehicleDriver:
    def __init__(self, node, topic, speed=1.0):
        self.pub = node.advertise(topic, Twist)
        self.speed = speed
        self.running = True

    def send(self, linear_x=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.pub.publish(twist)

    def stop(self):
        self.send(0.0, 0.0)

    def drive_circle(self, radius_factor=0.5):
        while self.running:
            self.send(self.speed, self.speed * radius_factor)
            time.sleep(0.05)

    def drive_random(self):
        while self.running:
            fwd = random.uniform(0.3, 1.0) * self.speed
            turn = random.uniform(-1.0, 1.0)
            duration = random.uniform(1.0, 4.0)
            t = 0
            while t < duration and self.running:
                self.send(fwd, turn)
                time.sleep(0.05)
                t += 0.05
            if random.random() < 0.2 and self.running:
                self.stop()
                time.sleep(random.uniform(0.5, 1.5))

    def drive_square(self):
        while self.running:
            t = 0
            while t < 3.0 and self.running:
                self.send(self.speed, 0.0)
                time.sleep(0.05)
                t += 0.05
            self.stop()
            time.sleep(0.3)
            t = 0
            while t < 1.57 and self.running:
                self.send(0.0, 1.0)
                time.sleep(0.05)
                t += 0.05
            self.stop()
            time.sleep(0.3)


def main():
    node = Node()
    time.sleep(0.5)

    # Create drivers for each vehicle with different patterns
    vehicles = [
        {"name": "RED", "topic": "/ground_vehicle/cmd_vel", "pattern": "circle", "speed": 1.0},
        {"name": "BLUE", "topic": "/ground_vehicle_blue/cmd_vel", "pattern": "random", "speed": 0.8},
        {"name": "GREEN", "topic": "/ground_vehicle_green/cmd_vel", "pattern": "square", "speed": 1.2},
    ]

    drivers = []
    threads = []

    print("=" * 50)
    print("  MULTI-VEHICLE DRIVER")
    print("=" * 50)

    for v in vehicles:
        driver = VehicleDriver(node, v["topic"], v["speed"])
        drivers.append(driver)

        pattern_func = {
            "circle": driver.drive_circle,
            "random": driver.drive_random,
            "square": driver.drive_square,
        }[v["pattern"]]

        t = threading.Thread(target=pattern_func, daemon=True)
        threads.append(t)
        print(f"  {v['name']}: {v['pattern']} at speed {v['speed']}")

    print("=" * 50)
    print("  Press Ctrl+C to stop all vehicles")
    print("=" * 50)

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping all vehicles...")
        for driver in drivers:
            driver.running = False
            driver.stop()
        time.sleep(0.5)
        print("Done!")


if __name__ == "__main__":
    main()
