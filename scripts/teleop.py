#!/usr/bin/env python3
"""
Keyboard teleop for the quadrotor drone.
WASD = forward/back/left/right
Q/E  = up/down
Arrow keys = yaw
"""
import sys
import tty
import termios
from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist

SPEED = 1.0   # m/s
YAW_SPEED = 1.0

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def main():
    node = Node()
    pub = node.advertise("/drone/cmd_vel", Twist)

    # Enable the drone controller first
    from gz.msgs10.boolean_pb2 import Boolean
    enable_pub = node.advertise("/drone/enable", Boolean)
    msg = Boolean()
    msg.data = True
    enable_pub.publish(msg)

    print("Drone Teleop — WASD to move, Q/E for altitude, arrows to yaw, X to quit")

    while True:
        key = get_key()
        twist = Twist()

        if key == 'w':   twist.linear.x =  SPEED
        elif key == 's': twist.linear.x = -SPEED
        elif key == 'a': twist.linear.y =  SPEED
        elif key == 'd': twist.linear.y = -SPEED
        elif key == 'q': twist.linear.z =  SPEED
        elif key == 'e': twist.linear.z = -SPEED
        elif key == '\x1b':  # arrow keys
            next1, next2 = get_key(), get_key()
            if next2 == 'C':  twist.angular.z = -YAW_SPEED  # right arrow
            elif next2 == 'D': twist.angular.z =  YAW_SPEED  # left arrow
        elif key == 'x':
            print("Exiting.")
            break

        pub.publish(twist)

if __name__ == '__main__':
    main()
