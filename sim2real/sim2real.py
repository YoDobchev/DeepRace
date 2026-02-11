from __future__ import annotations

import argparse
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from stable_baselines3 import DQN


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def now() -> float:
    return time.time()


def apply_pair_deadzone(
    l: float,
    r: float,
    *,
    motor_min: float,
    motor_max: float,
    eps: float,
) -> tuple[float, float]:
    m = max(abs(l), abs(r))
    if m < eps:
        return 0.0, 0.0

    m = clamp(m, 0.0, 1.0)
    m_out = motor_min + (motor_max - motor_min) * m
    scale = m_out / max(1e-6, m)

    l2 = clamp(l * scale, -motor_max, motor_max)
    r2 = clamp(r * scale, -motor_max, motor_max)
    return l2, r2


@dataclass
class IMUState:
    gz: float = 0.0
    t: float = 0.0
    esp_ip: Optional[str] = None


ACTION_TABLE = [
    (0.0, 0.0, 0),
    (0.4, 0.4, 0),
    (0.7, 0.7, 0),
    (1.0, 1.0, 0),
    (0.3, 0.7, 0),
    (0.0, 0.7, 0),
    (0.7, 0.3, 0),
    (0.7, 0.0, 0),
    (0.0, 0.0, 1),
]


class MJPEGCamera:
    def __init__(self, url: str, timeout_s: float = 5.0):
        self.url = url
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._t_frame: float = 0.0
        self._stop = False
        self._thr: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop = True
        if self._thr is not None:
            self._thr.join(timeout=1.0)

    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._t_frame

    def _run(self) -> None:
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"

        headers = {"User-Agent": "Mozilla/5.0"}
        while not self._stop:
            resp = None
            try:
                resp = requests.get(
                    self.url,
                    stream=True,
                    headers=headers,
                    timeout=(self.timeout_s, self.timeout_s),
                )

                buf = b""
                for chunk in resp.iter_content(chunk_size=8192):
                    if self._stop:
                        break
                    if not chunk:
                        continue

                    buf += chunk

                    end = buf.rfind(EOI)
                    if end != -1:
                        start = buf.rfind(SOI, 0, end)
                        if start != -1 and end > start:
                            jpg = buf[start: end + 2]
                            buf = buf[end + 2:]

                            arr = np.frombuffer(jpg, dtype=np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is None:
                                continue
                            with self._lock:
                                self._frame = frame
                                self._t_frame = now()

                try:
                    resp.close()
                except Exception:
                    pass
            except Exception:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                time.sleep(0.2)


class IMUReceiver:
    def __init__(self, bind_ip: str, imu_port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, imu_port))
        self.sock.setblocking(False)
        self.state = IMUState()

    def poll(self) -> IMUState:
        while True:
            try:
                data, addr = self.sock.recvfrom(2048)
            except BlockingIOError:
                break
            except Exception:
                break

            line = data.decode("utf-8", errors="replace").strip()
            parts = line.split(",")

            try:
                if len(parts) == 1:
                    self.state.gz = float(parts[0])
                elif len(parts) == 7:
                    self.state.gz = float(parts[5])
                else:
                    continue
            except ValueError:
                continue

            self.state.t = now()
            self.state.esp_ip = addr[0]

        return self.state


def extract_line_features(
    frame_bgr: np.ndarray,
    *,
    roi_y_frac: float = 0.45,
    thresh: int = 90,
    min_points: int = 8,
) -> Tuple[Optional[float], Optional[float], float, np.ndarray]:
    dbg = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    y0 = int(h * roi_y_frac)
    roi = frame_bgr[y0:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.putText(dbg, "NO LINE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return None, None, 0.0, dbg

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    roi_area = float(mask.shape[0] * mask.shape[1])
    conf = clamp(area / max(1.0, roi_area), 0.0, 1.0)

    tape = np.zeros_like(mask)
    cv2.drawContours(tape, [c], -1, 255, thickness=cv2.FILLED)

    roi_h, roi_w = tape.shape[:2]
    ys = np.linspace(roi_h - 5, 5, 25).astype(int)

    pts_y = []
    pts_x = []
    for yy in ys:
        xs = np.where(tape[yy, :] > 0)[0]
        if xs.size < 10:
            continue
        pts_y.append(float(yy))
        pts_x.append(float(xs.mean()))

    if len(pts_x) < min_points:
        cv2.putText(dbg, "LINE WEAK", (0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return None, None, conf * 0.3, dbg

    yv = np.array(pts_y, dtype=np.float32)
    xv = np.array(pts_x, dtype=np.float32)
    a, b, c0 = np.polyfit(yv, xv, 2)

    y_ref = float(roi_h - 5)
    x_ref = float(a * y_ref * y_ref + b * y_ref + c0)

    cx = roi_w * 0.5
    e_y_norm = float((x_ref - cx) / max(1.0, cx))

    dxdy = float(2.0 * a * y_ref + b)
    theta = float(np.arctan(dxdy))

    for yy, xx in zip(pts_y, pts_x):
        cv2.circle(dbg, (int(xx), int(yy) + y0), 3, (0, 255, 255), -1)

    curve_pts = []
    for yy in range(5, roi_h, 6):
        xx = a * yy * yy + b * yy + c0
        curve_pts.append((int(xx), int(yy) + y0))
    for i in range(1, len(curve_pts)):
        cv2.line(dbg, curve_pts[i - 1], curve_pts[i], (255, 0, 0), 2)

    cv2.circle(dbg, (int(x_ref), int(y_ref) + y0), 6, (0, 255, 0), -1)
    cv2.line(dbg, (int(cx), y0), (int(cx), h), (200, 200, 200), 1)

    return e_y_norm, theta, conf, dbg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="Path to SB3 .zip model")
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--esp-cam-ip", type=str, required=True,
                    help="ESP32-CAM IP (e.g. 192.168.100.6)")
    ap.add_argument("--cam-port", type=int, default=81)
    ap.add_argument("--cam-path", type=str, default="/stream")

    ap.add_argument("--imu-port", type=int, default=9000)
    ap.add_argument("--cmd-port", type=int, default=9001)
    ap.add_argument("--bind-ip", type=str, default="0.0.0.0")
    ap.add_argument("--esp-cmd-ip", type=str, default="",
                    help="(Optional) Force command IP; otherwise learned from IMU sender")

    ap.add_argument("--send-hz", type=float, default=50.0)

    ap.add_argument("--max-speed", type=float, default=1.0,
                    help="Scale motor outputs (0..1)")
    ap.add_argument("--motor-min", type=float, default=0.70,
                    help="Minimum non-zero motor command needed to move (0..1)")
    ap.add_argument("--motor-eps", type=float, default=0.03,
                    help="Treat |cmd| < eps as zero")
    ap.add_argument("--motor-max", type=float, default=1.00,
                    help="Hard cap on motor command (0..1)")

    ap.add_argument("--track-width", type=float, default=0.25)
    ap.add_argument("--ey-sign", type=float, default=1.0,
                    help="Flip sign if steering is backwards (+1 or -1)")
    ap.add_argument("--gyro-units", type=str,
                    default="deg", choices=["deg", "rad"])

    ap.add_argument("--scale", type=float, default=0.5,
                    help="Resize factor for vision+display (0<scale<=1)")
    ap.add_argument("--roi-y-frac", type=float, default=0.45)
    ap.add_argument("--thresh", type=int, default=90)
    ap.add_argument("--min-conf", type=float, default=0.002)

    ap.add_argument("--imu-timeout", type=float, default=0.5)
    ap.add_argument("--cam-timeout", type=float, default=0.5)
    ap.add_argument("--no-line-timeout", type=float, default=0.4)

    ap.add_argument("--corner-theta", type=float, default=0.35,
                    help="rad (~0.35 = 20deg) force hard turn")
    ap.add_argument("--theta-gain", type=float, default=3.0,
                    help="map theta -> kappa in sim obs")
    ap.add_argument("--speed-min-scale", type=float, default=0.35)
    ap.add_argument("--theta-slow", type=float, default=1.2)
    ap.add_argument("--ey-slow", type=float, default=0.8)

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--print-hz", type=float, default=5.0,
                    help="limit console prints")

    args = ap.parse_args()

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    model = DQN.load(args.model, device=args.device)

    cam_url = f"http://{args.esp_cam_ip}:{args.cam_port}{args.cam_path}"
    cam = MJPEGCamera(cam_url)
    cam.start()

    imu = IMUReceiver(args.bind_ip, args.imu_port)

    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ey_max = args.track_width * 2.0
    send_period = 1.0 / max(1e-6, args.send_hz)
    last_send = 0.0

    last_good_line_t = 0.0

    last_print = 0.0
    print_period = 1.0 / max(1e-6, args.print_hz)

    def send_drive(ip: str, left: float, right: float) -> None:
        msg = f"{right:.3f},{left:.3f}\n".encode("utf-8")
        sock_tx.sendto(msg, (ip, args.cmd_port))

    def safe_stop(ip: Optional[str]) -> None:
        if not ip:
            return
        try:
            send_drive(ip, 0.0, 0.0)
        except Exception:
            pass

    print(f"camera URL: {cam_url}")

    if args.show:
        cv2.namedWindow("sim2real debug", cv2.WINDOW_NORMAL)

    try:
        while True:
            t = now()

            st = imu.poll()
            imu_age = t - st.t
            imu_ok = imu_age <= args.imu_timeout

            frame, t_frame = cam.get_latest()
            cam_age = t - t_frame
            cam_ok = (frame is not None) and (cam_age <= args.cam_timeout)

            cmd_ip = args.esp_cmd_ip.strip() or (st.esp_ip or "")
            have_ip = bool(cmd_ip)

            left_out = 0.0
            right_out = 0.0

            show_img = None
            status_line = ""

            eyn = None
            theta = None
            conf = 0.0
            dbg = None

            if cam_ok and frame is not None:
                scale = clamp(args.scale, 0.1, 1.0)
                if scale != 1.0:
                    frame_small = cv2.resize(
                        frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                else:
                    frame_small = frame

                eyn, theta, conf, dbg = extract_line_features(
                    frame_small,
                    roi_y_frac=args.roi_y_frac,
                    thresh=args.thresh,
                )

                show_img = dbg
            else:
                show_img = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(show_img, "WAITING FOR CAMERA...", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

            line_ok = (eyn is not None) and (
                theta is not None) and (conf >= args.min_conf)

            if imu_ok and cam_ok and line_ok and have_ip:
                last_good_line_t = t

                e_y = float(args.ey_sign * eyn * ey_max)

                kappa = float(clamp(theta * args.theta_gain, -3.0, 3.0))

                yaw_rate = float(st.gz)
                if args.gyro_units == "deg":
                    yaw_rate = yaw_rate * (np.pi / 180.0)
                yaw_rate = float(clamp(yaw_rate, -20.0, 20.0))

                obs = np.array([e_y, kappa, yaw_rate], dtype=np.float32)

                action, _ = model.predict(obs, deterministic=True)
                action_i = int(action)

                if abs(theta) > args.corner_theta:
                    action_i = 5 if theta > 0 else 7

                lcmd, rcmd, brake = ACTION_TABLE[action_i]

                left_out = float(lcmd * args.max_speed)
                right_out = float(rcmd * args.max_speed)

                speed_scale = clamp(
                    1.0 - args.theta_slow *
                    abs(theta) - args.ey_slow * abs(eyn),
                    args.speed_min_scale,
                    1.0,
                )
                left_out *= speed_scale
                right_out *= speed_scale

                if brake:
                    left_out *= 0.35
                    right_out *= 0.35

                left_out, right_out = apply_pair_deadzone(
                    left_out, right_out,
                    motor_min=args.motor_min,
                    motor_max=args.motor_max,
                    eps=args.motor_eps,
                )

                status_line = f"OK  a={action_i}  L={left_out:+.3f} R={right_out:+.3f}  conf={conf:.4f}"
                if (t - last_print) >= print_period:
                    print(status_line)
                    last_print = t

            else:
                if have_ip and (t - last_good_line_t) > args.no_line_timeout:
                    left_out, right_out = 0.0, 0.0

                reasons = []
                if not have_ip:
                    reasons.append("no_ip")
                if not cam_ok:
                    reasons.append("cam_timeout")
                if not imu_ok:
                    reasons.append("imu_timeout")
                if cam_ok and not line_ok:
                    reasons.append("no_line")
                status_line = "STOP (" + ", ".join(reasons) + ")"

            if have_ip and (t - last_send) >= send_period:
                try:
                    send_drive(cmd_ip, right_out, left_out)
                except Exception:
                    pass
                last_send = t

            if args.show and show_img is not None:
                cv2.putText(show_img, status_line, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(show_img, f"age={imu_age:.2f}s  ip={cmd_ip or '(none)'}",
                            (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if cam_ok:
                    cv2.putText(show_img, f"eyn={eyn if eyn is not None else 'None'}  theta={theta if theta is not None else 'None'}  conf={conf:.4f}",
                                (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("sim2real debug", show_img)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        cmd_ip = args.esp_cmd_ip.strip() or (imu.state.esp_ip or "")
        if cmd_ip:
            safe_stop(cmd_ip)
            time.sleep(0.05)
            safe_stop(cmd_ip)
        cam.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sock_tx.close()


if __name__ == "__main__":
    main()
