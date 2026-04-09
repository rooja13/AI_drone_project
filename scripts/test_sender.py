import json
import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.1", 5005)

try:
    while True:
        cmd = {
            "vx": 0.2,
            "vy": 0.0,
            "vz": 0.0,
            "yaw_rate": 0.0
        }
        sock.sendto(json.dumps(cmd).encode("utf-8"), addr)
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
