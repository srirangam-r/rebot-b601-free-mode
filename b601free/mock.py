"""Fake hardware for --mock: a motorbridge-compatible CAN controller with simple motor physics, a
FashionStar-compatible leader servo bus playing a scripted motion, and synthetic / replayed cameras.

The fakes sit *below* b601free/b601.py and b601free/leader.py, so the full driver code (MIT gains, gripper
impedance, temperature checks, gripper zero detection, safe_zero, unwrap/clamp, ...) runs in mock mode.
"""

from __future__ import annotations

import glob
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import JOINTS


# ---------------------------------------------------------------------- follower (motorbridge API subset)
@dataclass(frozen=True)
class FakeMotorState:
    can_id: int
    arbitration_id: int
    status_code: int
    pos: float
    vel: float
    torq: float
    t_mos: float
    t_rotor: float


_INERTIA = {"rs-06": 0.08, "rs-00": 0.01}
_TMAX = {"rs-06": 36.0, "rs-00": 14.0}


class FakeMotor:
    def __init__(self, ctl: "FakeController", idx: int, can_id: int, model: str):
        self.ctl, self.i, self.can_id, self.model = ctl, idx, can_id, model
        self.closed = False

    def _check(self):
        if self.closed:
            raise RuntimeError("motor handle is closed")

    def enable(self):
        self._check(); self.ctl.enabled[self.i] = True

    def disable(self):
        self._check(); self.ctl.enabled[self.i] = False

    def clear_error(self):
        self._check()

    def close(self):
        self.closed = True

    def set_zero_position(self):
        self._check(); self.ctl.pos[self.i] = 0.0

    def ensure_mode(self, mode, timeout_ms: int = 1000):
        self._check(); self.ctl.mode[self.i] = int(mode)

    def send_mit(self, pos, vel, kp, kd, tau):
        self._check()
        self.ctl.integrate()
        self.ctl.cmd[self.i] = (float(pos), float(vel), float(kp), float(kd), float(tau))
        self.ctl.n_cmds += 1

    def request_feedback(self):
        self._check(); self.ctl.requested[self.i] = True

    def get_state(self):
        self._check()
        c = self.ctl
        if not c.has_state[self.i]:
            return None
        return FakeMotorState(self.can_id, 0xFD, 0, float(c.pos_fb[self.i]), float(c.vel_fb[self.i]),
                              float(c.torq_fb[self.i]), float(c.temp_c[self.i]), float(c.temp_c[self.i]))


class FakeController:
    """1 kHz explicit integration of I*qdd = tau_motor - b*qd with hard stops, run on demand up to now."""

    DT = 0.001

    def __init__(self, channel: str = "can0"):
        self.channel = channel
        self.motors: list[FakeMotor] = []
        self.t_last = time.perf_counter()
        self.lock = threading.Lock()
        self.n_cmds = 0
        self.closed = False

    @classmethod
    def factory(cls, cfg):
        ctl = cls(cfg.port)
        ctl.limits = {n: (math.radians(lo) - 0.05, math.radians(hi) + 0.05) for n, (lo, hi) in cfg.joint_limits.items()}
        return ctl

    def add_robstride_motor(self, motor_id: int, feedback_id: int, model: str) -> FakeMotor:
        i = len(self.motors)
        m = FakeMotor(self, i, motor_id, model)
        self.motors.append(m)
        n = len(self.motors)
        self.pos = np.zeros(n); self.vel = np.zeros(n); self.torq = np.zeros(n)
        self.pos[-1:] = math.radians(0.6) if n == 7 else 0.0  # gripper starts slightly open
        self.pos_fb, self.vel_fb, self.torq_fb = self.pos.copy(), self.vel.copy(), self.torq.copy()
        self.inertia = np.array([_INERTIA[mm.model] for mm in self.motors])
        if n == 7:
            self.inertia[6] = 0.004
        self.tmax = np.array([_TMAX[mm.model] for mm in self.motors])
        self.enabled = np.zeros(n, bool); self.mode = np.zeros(n, int)
        self.requested = np.zeros(n, bool); self.has_state = np.zeros(n, bool)
        self.cmd = [(0.0, 0.0, 0.0, 0.0, 0.0)] * n
        self.temp_c = np.full(n, 38.0)
        return m

    def _bounds(self):
        lo = np.full(len(self.motors), -np.inf); hi = np.full(len(self.motors), np.inf)
        for i, name in enumerate(JOINTS[: len(self.motors)]):
            if hasattr(self, "limits") and name in self.limits:
                lo[i], hi[i] = self.limits[name]
        if len(self.motors) == 7:
            lo[6] = 0.0  # gripper closed hard stop at its zero
        return lo, hi

    def integrate(self):
        with self.lock:
            now = time.perf_counter()
            elapsed = min(now - self.t_last, 0.2)
            self.t_last = now
            steps = int(elapsed / self.DT)
            if steps <= 0:
                return
            p_des = np.array([c[0] for c in self.cmd]); v_des = np.array([c[1] for c in self.cmd])
            kp = np.array([c[2] for c in self.cmd]); kd = np.array([c[3] for c in self.cmd])
            ff = np.array([c[4] for c in self.cmd])
            on = self.enabled.astype(float)
            lo, hi = self._bounds()
            for _ in range(steps):
                tau = on * np.clip(kp * (p_des - self.pos) + kd * (v_des - self.vel) + ff, -self.tmax, self.tmax)
                acc = (tau - 0.05 * self.vel) / self.inertia
                self.vel += acc * self.DT
                self.pos += self.vel * self.DT
                hit = (self.pos < lo) | (self.pos > hi)
                self.pos = np.clip(self.pos, lo, hi)
                self.vel[hit] = 0.0
            self.torq = tau

    def poll_feedback_once(self):
        if self.closed:
            raise RuntimeError("controller is closed")
        self.integrate()
        r = self.requested
        noise = np.random.default_rng().normal(0, 1e-4, len(r))
        self.pos_fb[r] = self.pos[r] + noise[r]
        self.vel_fb[r] = self.vel[r]
        self.torq_fb[r] = self.torq[r]
        self.has_state |= r
        self.requested[:] = False

    def enable_all(self):
        self.integrate(); self.enabled[:] = True

    def disable_all(self):
        self.integrate(); self.enabled[:] = False

    def shutdown(self):
        self.disable_all()

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------- leader (FashionStarServo API subset)
@dataclass
class FakeServoMonitor:
    id: int
    angle_deg: float
    reliable: bool = True


class FakeServoBus:
    """Scripted 'pick-and-place-like' leader motion in leader-frame degrees. wrist_roll is reported with an
    extra +360 deg turn to exercise the plugin's multi-turn unwrap."""

    def __init__(self, port: str, baudrate: int = 1_000_000):
        self.port, self.baudrate = port, baudrate
        self.t0 = time.perf_counter()
        self.is_open = True

    @classmethod
    def factory(cls, cfg):
        return cls(cfg.port, cfg.baudrate)

    def ping(self, servo_id: int) -> bool:
        return 0 <= servo_id <= 6

    def unlock(self, servo_id: int):
        pass

    def reset_multi_turn(self, servo_id: int):
        pass

    def set_origin_point(self, servo_id: int):
        pass

    def close(self):
        self.is_open = False

    @staticmethod
    def pose(t: float) -> np.ndarray:
        r = min(1.0, t / 2.0)  # ease in from the zero pose
        w = 2 * math.pi
        grip = 20 + 20 * math.tanh(3 * math.sin(w * t / 6))  # leader 0..40 deg -> follower 0..240 motor deg
        return np.array([
            25 * math.sin(w * t / 8) * r,
            (45 + 20 * math.sin(w * t / 6)) * r,
            -(70 + 25 * math.sin(w * t / 6 + 0.5)) * r,
            15 * math.sin(w * t / 5) * r,
            10 * math.sin(w * t / 7) * r,
            30 * math.sin(w * t / 9) * r,
            grip * r,
        ])

    def sync_monitor(self, servo_ids):
        a = self.pose(time.perf_counter() - self.t0)
        a[5] += 360.0
        return {i: FakeServoMonitor(i, float(a[i])) for i in servo_ids}


# ---------------------------------------------------------------------- cameras
class MockCamera:
    """Same interface as cameras.Camera. source: None -> synthetic frames (moving blobs + noise);
    a .npz episode or a directory of ep_*.npz -> replays that episode's `key` view ("front"/"wrist")."""

    def __init__(self, name: str, key: str, frames_path: str | None = None, fps: float = 30.0, size: int = 256):
        self.name, self.key, self.fps, self.size = name, key, fps, size
        self.t0 = time.perf_counter()
        self.frames = None
        if frames_path:
            p = Path(frames_path)
            files = sorted(glob.glob(str(p / "**" / "ep_*.npz"), recursive=True)) if p.is_dir() else [str(p)]
            if not files:
                raise FileNotFoundError(f"no ep_*.npz under {p}")
            self.frames = np.load(files[0])[key]
        self.rng = np.random.default_rng(0 if key == "front" else 1)
        yy, xx = np.mgrid[0:size, 0:size]
        self._yy, self._xx = yy, xx
        self.started = False

    def start(self):
        self.started = True
        return self

    def read(self):
        n = int((time.perf_counter() - self.t0) * self.fps)
        if self.frames is not None:
            return self.frames[n % len(self.frames)], 1.0 / self.fps / 2
        t = n / self.fps
        img = np.full((self.size, self.size, 3), (90, 110, 80) if self.key == "front" else (70, 70, 90), np.uint8)
        cx = self.size / 2 + 70 * math.sin(t * 0.9 + (0 if self.key == "front" else 1.5))
        cy = self.size / 2 + 50 * math.cos(t * 0.7)
        blob = (self._xx - cx) ** 2 + (self._yy - cy) ** 2 < 28 ** 2
        img[blob] = (200, 40, 40)
        img[200:230, 30:90] = (40, 180, 60)
        img = np.clip(img.astype(np.int16) + self.rng.integers(-8, 9, img.shape, dtype=np.int16), 0, 255)
        return img.astype(np.uint8), 1.0 / self.fps / 2

    def stop(self):
        self.started = False
