import socket
import time

IMU_PORT = 9000
CMD_PORT = 9001

sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_rx.bind(("0.0.0.0", IMU_PORT))
sock_rx.settimeout(1.0)

sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

esp_ip = None


def read_imu():
    global esp_ip
    data, addr = sock_rx.recvfrom(2048)
    esp_ip = addr[0]
    line = data.decode("utf-8", errors="replace").strip()
    parts = line.split(",")
    if len(parts) != 7:
        return None
    return tuple(map(float, parts))


def send_drive(left, right):
    if esp_ip is None:
        return
    msg = f"{left:.3f},{right:.3f}\n".encode("utf-8")
    sock_tx.sendto(msg, (esp_ip, CMD_PORT))


last = time.time()
count = 0

while True:
    try:
        sample = read_imu()
        if sample:
            count += 1
            print("IMU:", sample)
            send_drive(1, 0.7)
            time.sleep(2)
            send_drive(0.0, 0.0)
            time.sleep(2)

        now = time.time()
        if now - last >= 1.0:
            print("IMU Hz:", count)
            count = 0
            last = now

    except socket.timeout:
        pass
