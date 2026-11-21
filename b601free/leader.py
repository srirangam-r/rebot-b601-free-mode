"""reBot Arm 102 leader reader, ported from Seeed's lerobot_teleoperator_rebot_arm_102 (v1.0.1).

FashionStar UART servos via `motorbridge_smart_servo`, 1 Mbaud on /dev/ttyUSB0. get_action() is the
plugin's: sync_monitor -> angle_deg -> unwrap multi-turn into the +-180 window around the joint-range
centre -> clamp to joint_ranges. Values are leader/action-frame degrees.

Mapping to follower targets (what `lerobot-teleoperate` does with the default identity processors):
    follower_motor_deg = clip(leader_deg * joint_directions[j], joint_limits[j])   (b601.action_to_motor)
so the gripper is x6 (leader 0..45 deg -> follower motor 0..270 deg) and elbow/wrist_flex/wrist_yaw flip.

Addition: connect() does one read and fails if any servo never answered; the plugin would otherwise
fall back to its initial 0.0 positions, i.e. command the follower to the zero pose.

Usage:  python -m b601free.leader [--mock]   # print leader angles and the mapped follower targets
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .common import JOINTS, calibration_path, default_calibration_entry, fmt_vec, load_calibration, \
    save_calibration, setup_logging

logger = logging.getLogger(__name__)
MEDIUM_TIMEOUT_SEC = 0.01


@dataclass
class LeaderConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 1_000_000
    id: str = "rebot_arm_102_leader"  # calibration file id (lerobot --teleop.id)
    calibration_dir: str | None = None
    joint_ids: dict[str, int] = field(default_factory=lambda: {
        "shoulder_pan": 0, "shoulder_lift": 1, "elbow_flex": 2, "wrist_flex": 3,
        "wrist_yaw": 4, "wrist_roll": 5, "gripper": 6})
    joint_ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "shoulder_pan": (-150.0, 150.0), "shoulder_lift": (-1.0, 170.0), "elbow_flex": (-200.0, 1.0),
        "wrist_flex": (-80.0, 90.0), "wrist_yaw": (-90.0, 90.0), "wrist_roll": (-130.0, 130.0),
        "gripper": (-0.0, 270.0)})


class Leader:
    name = "rebot_arm_102_leader"

    def __init__(self, cfg: LeaderConfig, bus_factory: Callable[[LeaderConfig], Any] | None = None):
        self.cfg = self.config = cfg
        self._bus_factory = bus_factory
        self.bus = None
        self.motor_names = list(cfg.joint_ids.keys())
        if self.motor_names != JOINTS:
            raise ValueError(f"joint order must be {JOINTS}")
        self._last_raw_positions: dict[str, float] = {}
        self.consecutive_failures = 0
        self.calibration_fpath = calibration_path("teleoperators", self.name, cfg.id, cfg.calibration_dir)
        self.calibration = load_calibration(self.calibration_fpath)
        self._validate_config()

    def __repr__(self):
        return f"{self.cfg.id} RebotArm102Leader"

    def _validate_config(self) -> None:
        required = set(self.cfg.joint_ids)
        if set(self.cfg.joint_ranges) != required:
            raise ValueError(f"joint_ranges keys must match joint_ids keys: {sorted(required)}")
        for name, rng in self.cfg.joint_ranges.items():
            if len(rng) != 2 or rng[0] > rng[1]:
                raise ValueError(f"joint_ranges[{name!r}] must be [min, max] with min <= max.")

    @property
    def is_connected(self) -> bool:
        return self.bus is not None

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and set(self.calibration) == set(self.motor_names)

    def _open_bus(self):
        if self._bus_factory is not None:
            return self._bus_factory(self.cfg)
        from motorbridge_smart_servo import FashionStarServo
        return FashionStarServo(self.cfg.port, baudrate=self.cfg.baudrate)

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")
        logger.info("Connecting leader on %s...", self.cfg.port)
        bus = self._open_bus()
        try:
            for motor_name, motor_id in self.cfg.joint_ids.items():
                if not bus.ping(motor_id):
                    raise RuntimeError(f"Servo not found for {motor_name} (id={motor_id}).")
                self._last_raw_positions[motor_name] = 0.0
            self.bus = bus
            if not self.is_calibrated and calibrate:
                logger.info("No (complete) calibration file at %s", self.calibration_fpath)
                self.calibrate()
            self.configure()
            self._last_raw_positions = self._read_raw_positions()  # NEW: must answer once
        except BaseException:
            bus.close()
            self.bus = None
            raise
        logger.info("%s connected.", self)

    def calibrate(self) -> None:
        """Plugin calibrate(): user puts the leader in its zero pose with the gripper closed, each servo
        stores its origin, then the default calibration json is written."""
        if self.calibration:
            user_input = input(f"Press ENTER to use provided calibration file associated with the id {self.cfg.id}, "
                               "or type 'c' and press ENTER to run calibration: ")
            if user_input.strip().lower() != "c":
                logger.info("Using calibration file associated with the id %s", self.cfg.id)
                return
        input("\nCalibration: Set Zero Position\n"
              "Please manually move the reBot Arm 102 to its zero pose and close the gripper.\n"
              "Press ENTER when ready...")
        self.calibration = {}
        for motor_name, motor_id in self.cfg.joint_ids.items():
            self.bus.unlock(motor_id)
            time.sleep(MEDIUM_TIMEOUT_SEC)
            self.bus.set_origin_point(motor_id)
            self.calibration[motor_name] = default_calibration_entry(motor_id)
        save_calibration(self.calibration_fpath, self.calibration)
        logger.info("Calibration saved to %s", self.calibration_fpath)

    def configure(self) -> None:
        for motor_id in self.cfg.joint_ids.values():
            self.bus.unlock(motor_id)
            time.sleep(MEDIUM_TIMEOUT_SEC)
        for motor_id in self.cfg.joint_ids.values():  # reset multi-turn counter per servo
            self.bus.reset_multi_turn(motor_id)

    def _read_raw_positions(self) -> dict[str, float]:
        result = self.bus.sync_monitor(list(self.cfg.joint_ids.values()))
        id_to_name = {v: k for k, v in self.cfg.joint_ids.items()}
        raw = {}
        for motor_id, m in result.items():
            name = id_to_name[motor_id]
            if m is None:
                raise RuntimeError(f"Servo {name} (id={motor_id}) has never responded.")
            raw[name] = m.angle_deg
        return raw

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _round_to_valid_range(value: float, min_value: float, max_value: float) -> tuple[float, int]:
        """Unwrap a multi-turn angle into [centre-180, centre+180]; bidirectional search over k*360."""
        center = (min_value + max_value) / 2.0
        low, high = center - 180.0, center + 180.0
        for k in range(4096):
            cand = value + k * 360.0
            if low <= cand <= high:
                return cand, k
            cand = value - k * 360.0
            if low <= cand <= high:
                return cand, k
        return value - round((value - center) / 360.0) * 360.0, 4096

    def get_action(self) -> dict[str, float]:
        """{"<joint>.pos": leader degrees}. On a bus error: log the plugin's EMERGENCY STOP warnings and
        reuse the last good raw positions (the follower then holds)."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        try:
            raw = self._read_raw_positions()
            self._last_raw_positions = raw
            self.consecutive_failures = 0
        except Exception as e:
            self.consecutive_failures += 1
            logger.error("Failed to read raw positions: %s", e)
            logger.warning("[EMERGENCY STOP] Please hold the follower arm and cut off the main power to the arms.")
            logger.warning("[EMERGENCY STOP] Break the teleoperation session and check the USB connection or "
                           "power of the leader arm.")
            raw = self._last_raw_positions
        out = {}
        for name in self.motor_names:
            lo, hi = map(float, self.cfg.joint_ranges[name])
            unwrapped, k = self._round_to_valid_range(raw[name], lo, hi)
            if k > 0:
                logger.debug("Servo %s wrapped %d*360: %.1f (raw %.1f)", name, k, unwrapped, raw[name])
            out[f"{name}.pos"] = self._clamp(unwrapped, lo, hi)
        return out

    def read(self) -> np.ndarray:
        a = self.get_action()
        return np.array([a[f"{n}.pos"] for n in JOINTS])

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        try:
            self.bus.close()
        finally:
            self.bus = None
        logger.info("%s disconnected.", self)


def add_leader_args(p: argparse.ArgumentParser):
    p.add_argument("--leader_port", default="/dev/ttyUSB0")
    p.add_argument("--leader_id", default="rebot_arm_102_leader", help="calibration id (lerobot --teleop.id)")


def make_leader(args) -> Leader:
    factory = None
    if args.mock:
        from .mock import FakeServoBus
        factory = FakeServoBus.factory
    lead = Leader(LeaderConfig(port=args.leader_port, id=args.leader_id), bus_factory=factory)
    if args.mock and not lead.is_calibrated:  # never prompt in mock mode
        lead.calibration = {n: default_calibration_entry(i) for n, i in lead.cfg.joint_ids.items()}
    return lead


def main():
    from .b601 import B601Config, action_to_motor
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_leader_args(p)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--calibrate", action="store_true", help="run the plugin calibration (set servo origins)")
    p.add_argument("--hz", type=float, default=10.0)
    args = p.parse_args()
    setup_logging()
    lead = make_leader(args)
    if args.calibrate:
        lead.calibration = {}
    lead.connect(calibrate=True)
    fcfg = B601Config()
    print("joint order:", " ".join(f"{j[:9]:>9}" for j in JOINTS))
    try:
        while True:
            a = lead.read()
            m = action_to_motor(a, fcfg)
            print(f"\rleader {fmt_vec(a)} | follower target {fmt_vec([m[j] for j in JOINTS])}   ", end="", flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print()
    finally:
        lead.disconnect()


if __name__ == "__main__":
    main()
