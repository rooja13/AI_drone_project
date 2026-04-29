#!/usr/bin/env python3
"""
Drive multiple realistic ground vehicles simultaneously.

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

    def drive_circle(self):
        while self.running:
            self.send(self.speed, self.speed * 0.5)
            time.sleep(0.05)

    def drive_circle_reverse(self):
        while self.running:
            self.send(self.speed, -self.speed * 0.5)
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

    def drive_figure_eight(self):
        while self.running:
            t = 0
            while t < 6.0 and self.running:
                self.send(self.speed, 0.5)
                time.sleep(0.05)
                t += 0.05
            t = 0
            while t < 6.0 and self.running:
                self.send(self.speed, -0.5)
                time.sleep(0.05)
                t += 0.05

    def drive_zigzag(self):
        while self.running:
            t = 0
            while t < 2.0 and self.running:
                self.send(self.speed, 0.8)
                time.sleep(0.05)
                t += 0.05
            t = 0
            while t < 2.0 and self.running:
                self.send(self.speed, -0.8)
                time.sleep(0.05)
                t += 0.05


def main():
    node = Node()
    time.sleep(0.5)

    vehicles = [
        {"name": "CAR 1 (Hatchback)",  "topic": "/realistic_vehicle/cmd_vel",   "pattern": "circle",         "speed": 1.0},
        {"name": "SUV 1",              "topic": "/realistic_vehicle_2/cmd_vel",  "pattern": "random",         "speed": 0.8},
        {"name": "PICKUP 1",           "topic": "/realistic_vehicle_3/cmd_vel",  "pattern": "square",         "speed": 1.2},
        {"name": "CAR 2 (Hatchback)",  "topic": "/model/car_2/cmd_vel",         "pattern": "figure_eight",   "speed": 0.9},
        {"name": "SUV 2",              "topic": "/model/suv_2/cmd_vel",         "pattern": "zigzag",         "speed": 1.0},
        {"name": "PICKUP 2",           "topic": "/model/pickup_2/cmd_vel",      "pattern": "circle_reverse", "speed": 0.7},
    ]

    drivers = []
    threads = []

    print("=" * 55)
    print("  REALISTIC MULTI-VEHICLE DRIVER (6 vehicles)")
    print("=" * 55)

    for v in vehicles:
        driver = VehicleDriver(node, v["topic"], v["speed"])
        drivers.append(driver)

        pattern_func = {
            "circle": driver.drive_circle,
            "circle_reverse": driver.drive_circle_reverse,
            "random": driver.drive_random,
            "square": driver.drive_square,
            "figure_eight": driver.drive_figure_eight,
            "zigzag": driver.drive_zigzag,
        }[v["pattern"]]

        t = threading.Thread(target=pattern_func, daemon=True)
        threads.append(t)
        print(f"  {v['name']}: {v['pattern']} @ speed {v['speed']}")

    print("=" * 55)
    print("  Press Ctrl+C to stop all vehicles")
    print("=" * 55)

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
