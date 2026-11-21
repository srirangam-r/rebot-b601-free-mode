"""Shared helpers: joint names, LeRobot-compatible calibration files, a fixed-rate loop timer."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:6]
GRIPPER = "gripper"
ROOT = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent   # per-robot files (poses, calibration, logs) live here, git-ignored


def calibration_root() -> Path:
    """Same resolution as lerobot.utils.constants.HF_LEROBOT_CALIBRATION."""
    hf_home = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"),
                                                                    "huggingface")))
    lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", Path(hf_home) / "lerobot")).expanduser()
    return Path(os.getenv("HF_LEROBOT_CALIBRATION", lerobot_home / "calibration")).expanduser()


def calibration_path(kind: str, device_name: str, device_id: str, calibration_dir: str | None = None) -> Path:
    """kind = "robots" | "teleoperators". Mirrors Robot/Teleoperator.__init__ in LeRobot:
    <calibration_dir or HF_LEROBOT_CALIBRATION/kind/name>/<id>.json"""
    d = Path(calibration_dir) if calibration_dir else calibration_root() / kind / device_name
    return d / f"{device_id}.json"


def load_calibration(path: Path) -> dict[str, dict]:
    """{motor: {id, drive_mode, homing_offset, range_min, range_max}} (LeRobot MotorCalibration fields).
    The Seeed plugins only test for the file's presence; none of these values is applied to positions."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_calibration(path: Path, calib: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:  # draccus.dump(..., indent=4) produces the same JSON
        json.dump(calib, f, indent=4)


def default_calibration_entry(motor_id: int) -> dict:
    """What both plugins' calibrate() writes for every motor."""
    return {"id": int(motor_id), "drive_mode": 0, "homing_offset": 0, "range_min": -90, "range_max": 90}


class RateLoop:
    """Fixed-rate loop: `wait()` sleeps until the next tick (sleep + short busy-wait for precision).
    If a tick overruns by more than one period we re-anchor instead of trying to catch up."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.next_t = time.perf_counter() + self.period
        self.overruns = 0
        self.last_dt = 0.0
        self._t_prev = time.perf_counter()

    def wait(self):
        now = time.perf_counter()
        remaining = self.next_t - now
        if remaining > 0.002:
            time.sleep(remaining - 0.0015)
        while time.perf_counter() < self.next_t:
            pass
        now = time.perf_counter()
        if now - self.next_t > self.period:
            self.overruns += 1
            self.next_t = now + self.period
        else:
            self.next_t += self.period
        self.last_dt, self._t_prev = now - self._t_prev, now

    def reset(self):
        self.next_t = time.perf_counter() + self.period
        self._t_prev = time.perf_counter()


def setup_logging(verbose: bool = False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")


def fmt_vec(v, prec: int = 1) -> str:
    return " ".join(f"{x:7.{prec}f}" for x in v)
