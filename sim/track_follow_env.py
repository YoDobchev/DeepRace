import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit


@dataclass
class Track:
    pts: np.ndarray
    s: np.ndarray
    heading: np.ndarray
    kappa: np.ndarray
    width: float


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def generate_random_track(
    rng: np.random.Generator,
    n_segments: int = 12,
    seg_len_range: Tuple[float, float] = (0.4, 1.0),
    kappa_range: Tuple[float, float] = (-1.2, 1.2),
    ds: float = 0.02,
    width: float = 0.25,
) -> Track:
    x, y, th = 0.0, 0.0, 0.0
    pts: list[tuple[float, float]] = []
    s_list: list[float] = []
    th_list: list[float] = []
    k_list: list[float] = []

    s_acc = 0.0
    for _ in range(n_segments):
        L = float(rng.uniform(*seg_len_range))
        k = float(rng.uniform(*kappa_range))
        if rng.random() < 0.35:
            k *= 0.25

        n = max(2, int(L / ds))
        for _i in range(n):
            th += k * ds
            x += math.cos(th) * ds
            y += math.sin(th) * ds
            s_acc += ds

            pts.append((x, y))
            s_list.append(s_acc)
            th_list.append(th)
            k_list.append(k)

    return Track(
        pts=np.asarray(pts, dtype=np.float32),
        s=np.asarray(s_list, dtype=np.float32),
        heading=np.asarray(th_list, dtype=np.float32),
        kappa=np.asarray(k_list, dtype=np.float32),
        width=width,
    )


class TapeLineFollowEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        dt: float = 0.05,
        wheel_base: float = 0.15,
        max_wheel_speed: float = 1.2,
        track_width: float = 0.25,
        sensor_noise: Optional[Dict[str, float]] = None,
        episode_seconds: float = 30.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = float(dt)
        self.L = float(wheel_base)
        self.vmax = float(max_wheel_speed)
        self.track_width = float(track_width)
        self.max_steps = int(episode_seconds / self.dt)
        self.last_idx = 0
        self.stall_count = 0
        self.stall_steps = 0
        self.stall_limit = int(2.0 / self.dt)
        self.v_eps = 0.02
        self.progress_eps = 1e-3
        self.time_penalty = 0.001
        self.best_s = 0.0
        self.last_best_s = 0.0
        self.progress_eps = 0.005
        self.stall_limit = int(1.5 / self.dt)

        # --- noise: removed accel ---
        if sensor_noise is None:
            sensor_noise = {
                "e_y": 0.003,
                "kappa": 0.03,
                "yaw_rate": 0.02,
            }
        self.noise: Dict[str, float] = sensor_noise

        self.action_table: List[Tuple[float, float, int]] = [
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

        self.action_space: spaces.Discrete = spaces.Discrete(
            len(self.action_table))

        # --- observation: now 3D (e_y, kappa, yaw_rate) ---
        ey_max = self.track_width * 2.0
        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([-ey_max, -3.0, -20.0], dtype=np.float32),
            high=np.array([ey_max, 3.0, 20.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._track: Optional[Track] = None

        # state
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.vl = 0.0
        self.vr = 0.0
        self.v = 0.0
        self.last_s = 0.0
        self.step_count = 0

        self._pg: Any = None
        self._screen: Any = None
        self._clock: Any = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._track = generate_random_track(
            rng=self._rng,
            n_segments=14,
            seg_len_range=(0.5, 1.1),
            kappa_range=(-1.4, 1.4),
            ds=0.02,
            width=self.track_width,
        )

        tr = self._track

        self.x, self.y = float(tr.pts[10, 0]), float(tr.pts[10, 1])
        self.th = float(tr.heading[10] + self._rng.normal(0, 0.15))
        self.x += float(self._rng.normal(0, 0.03))
        self.y += float(self._rng.normal(0, 0.03))
        self.stall_steps = 0

        self.vl = self.vr = self.v = 0.0
        self.step_count = 0

        idx, ey, kappa = self._measure_track_features()
        self.last_s = float(tr.s[idx])

        self.best_s = self.last_s
        self.last_best_s = self.best_s
        self.stall_steps = 0

        obs = self._make_obs(ey, kappa, yaw_rate=0.0)
        info = {"track_idx": idx, "s": self.last_s, "e_y": ey, "kappa": kappa}
        return obs, info

    def step(self, action: int):
        assert self._track is not None, "Call reset() first"
        tr = self._track

        left_cmd, right_cmd, brake = self.action_table[int(action)]

        vlt = left_cmd * self.vmax
        vrt = right_cmd * self.vmax
        tau = 0.10
        a = self.dt / max(1e-6, tau)
        self.vl += a * (vlt - self.vl)
        self.vr += a * (vrt - self.vr)
        if brake:
            self.vl *= 0.35
            self.vr *= 0.35

        v = 0.5 * (self.vl + self.vr)
        yaw_rate = (self.vr - self.vl) / max(1e-6, self.L)

        self.th = _wrap_pi(self.th + yaw_rate * self.dt)
        self.x += math.cos(self.th) * v * self.dt
        self.y += math.sin(self.th) * v * self.dt

        self.v = v

        idx, ey, kappa = self._measure_track_features()
        s_now = float(tr.s[idx])

        self.best_s = max(self.best_s, s_now)
        progress_best = self.best_s - self.last_best_s
        self.last_best_s = self.best_s

        reward = (
            2.0 * progress_best
            - 1.5 * abs(ey)
            - 0.05 * abs(yaw_rate)
            - self.time_penalty
        )

        half_w = 0.5 * tr.width
        off_track = abs(ey) > half_w
        reached_end = idx >= (len(tr.s) - 5)
        terminated = bool(off_track or reached_end)

        self.step_count += 1
        truncated = bool(self.step_count >= self.max_steps)

        if off_track:
            reward -= 2.0
        if reached_end:
            reward += 3.0

        if progress_best < self.progress_eps:
            self.stall_steps += 1
        else:
            self.stall_steps = 0

        stalled = self.stall_steps >= self.stall_limit
        if stalled and not terminated:
            truncated = True
            reward -= 2.0

        obs = self._make_obs(ey, kappa, yaw_rate=yaw_rate)
        info = {
            "track_idx": idx,
            "s": s_now,
            "progress_best": progress_best,
            "e_y": ey,
            "kappa": kappa,
            "yaw_rate": yaw_rate,
            "wheel_speeds": (self.vl, self.vr),
            "cmd": (left_cmd, right_cmd, brake),
            "stalled": stalled,
        }

        if self.render_mode is not None:
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _measure_track_features(self) -> Tuple[int, float, float]:
        assert self._track is not None
        tr = self._track
        pts = tr.pts

        dx = pts[:, 0] - self.x
        dy = pts[:, 1] - self.y
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))

        th_ref = float(tr.heading[idx])
        nx, ny = -math.sin(th_ref), math.cos(th_ref)

        vx = self.x - float(pts[idx, 0])
        vy = self.y - float(pts[idx, 1])
        e_y = vx * nx + vy * ny

        kappa = float(tr.kappa[idx])

        e_y += float(self._rng.normal(0, self.noise["e_y"]))
        kappa += float(self._rng.normal(0, self.noise["kappa"]))

        return idx, e_y, kappa

    def _make_obs(self, e_y: float, kappa: float, yaw_rate: float) -> np.ndarray:
        yaw_rate = float(
            yaw_rate + self._rng.normal(0, self.noise["yaw_rate"]))
        obs = np.array([e_y, kappa, yaw_rate], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def render(self):
        if self.render_mode is None:
            return None
        assert self._track is not None

        if self._pg is None:
            import pygame
            self._pg = pygame
            pygame.init()
            self._screen = pygame.display.set_mode((900, 600))
            pygame.display.set_caption("TapeLineFollowEnv")
            self._clock = pygame.time.Clock()

        pygame = self._pg
        screen = self._screen
        clock = self._clock

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        screen.fill((18, 18, 22))

        tr = self._track
        pts = tr.pts

        minx, miny = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
        maxx, maxy = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
        pad = 0.3
        minx -= pad
        miny -= pad
        maxx += pad
        maxy += pad

        W, H = screen.get_width(), screen.get_height()
        sx = W / max(1e-6, (maxx - minx))
        sy = H / max(1e-6, (maxy - miny))
        s = 0.9 * min(sx, sy)

        ox = W * 0.5 - s * (minx + maxx) * 0.5
        oy = H * 0.5 + s * (miny + maxy) * 0.5

        def to_screen(px, py):
            return int(ox + s * px), int(oy - s * py)

        poly = [to_screen(float(p[0]), float(p[1])) for p in pts[::2]]
        if len(poly) >= 2:
            pygame.draw.lines(screen, (240, 240, 240), False, poly, 2)

        half_w = 0.5 * tr.width
        edgeL = []
        edgeR = []
        for i in range(0, len(pts), 6):
            th = float(tr.heading[i])
            nx, ny = -math.sin(th), math.cos(th)
            px, py = float(pts[i, 0]), float(pts[i, 1])
            edgeL.append(to_screen(px + nx * half_w, py + ny * half_w))
            edgeR.append(to_screen(px - nx * half_w, py - ny * half_w))
        if len(edgeL) >= 2:
            pygame.draw.lines(screen, (60, 60, 70), False, edgeL, 1)
            pygame.draw.lines(screen, (60, 60, 70), False, edgeR, 1)

        cx, cy = to_screen(self.x, self.y)
        th = self.th
        r = 12
        p1 = (cx + int(r * math.cos(th)), cy - int(r * math.sin(th)))
        p2 = (cx + int(r * math.cos(th + 2.5)),
              cy - int(r * math.sin(th + 2.5)))
        p3 = (cx + int(r * math.cos(th - 2.5)),
              cy - int(r * math.sin(th - 2.5)))
        pygame.draw.polygon(screen, (80, 200, 120), [p1, p2, p3])

        pygame.display.flip()
        clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(screen)
            return np.transpose(arr, (1, 0, 2))
        return None

    def close(self):
        if self._pg is not None:
            self._pg.quit()
        self._pg = None
        self._screen = None
        self._clock = None


if __name__ == "__main__":
    env = TapeLineFollowEnv(render_mode="human")
    env = TimeLimit(env, max_episode_steps=600)
    obs, info = env.reset()
    while True:
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            obs, info = env.reset()
