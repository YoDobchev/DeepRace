import socket
import time
import math
import pygame

IMU_PORT = 9000
CMD_PORT = 9001

sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_rx.bind(("0.0.0.0", IMU_PORT))
sock_rx.setblocking(False)

sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

esp_ip = None
last_rx_t = 0.0
last_imu = None


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def mix_diff_drive(throttle, steer):
    left = throttle - steer
    right = throttle + steer
    m = max(1.0, abs(left), abs(right))
    return left / m, right / m


def send_drive(left, right):
    global esp_ip
    if esp_ip is None:
        return
    msg = f"{left:.3f},{right:.3f}\n".encode("utf-8")
    sock_tx.sendto(msg, (esp_ip, CMD_PORT))


def poll_imu():
    global esp_ip, last_rx_t, last_imu
    while True:
        try:
            data, addr = sock_rx.recvfrom(2048)
        except BlockingIOError:
            break

        esp_ip = addr[0]
        last_rx_t = time.time()

        line = data.decode("utf-8", errors="replace").strip()
        parts = line.split(",")
        if len(parts) == 7:
            try:
                last_imu = tuple(map(float, parts))
            except ValueError:
                pass


pygame.init()
W, H = 720, 420
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("ESP32 Car Controller (UDP)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)


throttle = 0.0
steer = 0.0
max_speed = 0.7


THROTTLE_RATE = 1.6
STEER_RATE = 2.6
STEER_RETURN = 3.5
THROTTLE_RETURN = 1.8

SEND_HZ = 50.0
send_period = 1.0 / SEND_HZ
send_accum = 0.0

x, y = W * 0.5, H * 0.65
heading = -math.pi / 2
v_vis = 0.0

running = True
while running:
    dt = clock.tick(60) / 1000.0
    send_accum += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                throttle = 0.0
                steer = 0.0
            if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                max_speed = clamp(max_speed - 0.05, 0.1, 1.0)
            if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                max_speed = clamp(max_speed + 0.05, 0.1, 1.0)

    poll_imu()

    keys = pygame.key.get_pressed()

    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    if up and not down:
        throttle += THROTTLE_RATE * dt
    elif down and not up:
        throttle -= THROTTLE_RATE * dt
    else:
        if throttle > 0:
            throttle = max(0.0, throttle - THROTTLE_RETURN * dt)
        elif throttle < 0:
            throttle = min(0.0, throttle + THROTTLE_RETURN * dt)

    leftk = keys[pygame.K_LEFT] or keys[pygame.K_a]
    rightk = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    if rightk and not leftk:
        steer += STEER_RATE * dt
    elif leftk and not rightk:
        steer -= STEER_RATE * dt
    else:
        if steer > 0:
            steer = max(0.0, steer - STEER_RETURN * dt)
        elif steer < 0:
            steer = min(0.0, steer + STEER_RETURN * dt)

    throttle = clamp(throttle, -1.0, 1.0)
    steer = clamp(steer, -1.0, 1.0)

    now = time.time()
    imu_ok = (now - last_rx_t) < 2.0

    left_cmd, right_cmd = mix_diff_drive(throttle, steer)
    left_cmd *= max_speed
    right_cmd *= max_speed

    if not imu_ok:
        left_cmd = 0.0
        right_cmd = 0.0

    while send_accum >= send_period:
        send_drive(left_cmd, right_cmd)
        send_accum -= send_period

    target_v = throttle * 180.0
    v_vis += (target_v - v_vis) * min(1.0, 6.0 * dt)
    heading += steer * 1.8 * dt
    x += math.cos(heading) * v_vis * dt
    y += math.sin(heading) * v_vis * dt
    x = clamp(x, 20, W - 20)
    y = clamp(y, 20, H - 20)

    screen.fill((18, 18, 22))

    car_w, car_h = 18, 34
    car_surf = pygame.Surface((car_w, car_h), pygame.SRCALPHA)
    pygame.draw.rect(car_surf, (230, 230, 240),
                     (0, 0, car_w, car_h), border_radius=6)
    pygame.draw.rect(car_surf, (80, 200, 120),
                     (3, 3, car_w - 6, 10), border_radius=4)
    rot = pygame.transform.rotate(car_surf, -math.degrees(heading) - 90)
    rect = rot.get_rect(center=(x, y))
    screen.blit(rot, rect.topleft)

    hud = [
        f"ESP IP: {esp_ip if esp_ip else '(waiting for IMU UDP...)'}",
        f"IMU ok: {imu_ok}",
        f"Throttle: {throttle:+.2f}   Steer: {steer:+.2f}   Max speed: {max_speed:.2f}",
        f"Left cmd: {left_cmd:+.2f}   Right cmd: {right_cmd:+.2f}",
        "Controls: Arrows/WASD drive | +/- max speed | SPACE stop | Close window to stop",
    ]
    if last_imu:
        ax, ay, az, gx, gy, gz, tC = last_imu
        hud.append(
            f"IMU: ax={ax:+.3f} ay={ay:+.3f} az={az:+.3f}  gx={gx:+.3f} gy={gy:+.3f} gz={gz:+.3f}  T={tC:.1f}C")
        print(gz)

    y0 = 10
    for line in hud:
        txt = font.render(line, True, (235, 235, 245))
        screen.blit(txt, (10, y0))
        y0 += 22

    pygame.display.flip()

try:
    send_drive(0.0, 0.0)
    time.sleep(0.05)
    send_drive(0.0, 0.0)
except Exception:
    pass

pygame.quit()
