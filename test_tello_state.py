import socket
import time

TELLO_IP = "192.168.10.1"
CMD_PORT = 8889
STATE_PORT = 8890

cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cmd.bind(("", 8889))
cmd.settimeout(5)

state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state.bind(("", STATE_PORT))
state.settimeout(10)

print("Sending SDK command...")
cmd.sendto(b"command", (TELLO_IP, CMD_PORT))

try:
    data, addr = cmd.recvfrom(1024)
    print("Command response:", data, "from", addr)
except Exception as e:
    print("No command response:", e)

print("Waiting for state packet on UDP 8890...")
try:
    data, addr = state.recvfrom(4096)
    print("State packet from", addr)
    print(data.decode(errors="replace"))
except Exception as e:
    print("No state packet received:", e)
