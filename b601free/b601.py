"""Seeed reBot B601-RS follower driver, ported from the Seeed LeRobot plugin without LeRobot.

Source: lerobot_robot_seeed_b601 v1.0.5 (seeed_b601_follower.py + seeed_b601_rs_follower.py +
config_seeed_b601_rs_follower.py). All motor I/O goes through `motorbridge` exactly as in the plugin.
Only the RobStride (RS) variant is ported (MIT mode on all 7 motors).

UNITS: everything this module returns or accepts is in follower MOTOR-FRAME degrees (deg/s, N*m):
  * read()["q"] == plugin get_observation()["<joint>.pos"] == math.degrees(state.pos)
  * command(targets) takes motor-frame degrees, i.e. what the plugin's send_action() computes AFTER
    `position * joint_directions[joint]` and joint-limit clipping (and what send_action() returns).
  * action_to_motor() is that plugin mapping: leader/action-frame degrees -> motor-frame degrees.
Joint limits in the config are in the motor frame (the plugin clips after applying the direction).

Additions over the plugin (all on the safe side, documented in README):
  * per-step max-change clamp vs. the previous command (`max_step_deg`), on top of the plugin's
    optional `max_relative_target` (vs. present position, LeRobot's ensure_safe_goal_position).
  * connect() fails if a motor never answers (the plugin would silently report 0.0 for it).
  * hold(), read-only mode (torque off) and dry-run (`send=False`) helpers.
  * disconnect(): a Ctrl-C during safe_zero still disables torque; one motor failing to disable
    does not stop the others from being disabled.

Usage:
  python -m b601free.b601 read [--mock]          # torque OFF, print q (move the arm by hand)
  python -m b601free.b601 calibrate              # plugin calibrate(): set zero + write calibration json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

import numpy as np

from .common import (GRIPPER, JOINTS, calibration_path, default_calibration_entry, fmt_vec, load_calibration,
                     save_calibration, setup_logging)
from .gravity import DEFAULT_URDF

logger = logging.getLogger(__name__)

FOLLOWER_GRIPPER_MOTOR = GRIPPER
LONG_TIMEOUT_SEC = 0.1
MEDIUM_TIMEOUT_SEC = 0.01
MIT_MODE = 1  # motorbridge.Mode.MIT


class Overheat(KeyboardInterrupt):
    """The plugin raises KeyboardInterrupt('Overheat detected'); this subclass keeps that behaviour
    (control loops treat it like Ctrl-C) while letting callers print a specific message."""


@dataclass
class B601Config:
    port: str = "can0"
    can_adapter: str = "socketcan"  # "socketcan" | "damiao" (serial bridge)
    dm_serial_baud: int = 921600
    id: str = "follower1"  # calibration file id (lerobot --robot.id)
    calibration_dir: str | None = None
    disable_torque_on_disconnect: bool = True
    # Plugin option: cap |goal - present| in degrees (None = off, plugin default).
    max_relative_target: float | dict[str, float] | None = None
    # NEW: cap |goal - previous goal| per command, degrees (float, or dict per joint; None = off).
    max_step_deg: float | dict[str, float] | None = None

    # ---- config_seeed_b601_rs_follower.py defaults, verbatim
    mit_kp: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 50.0, "shoulder_lift": 150.0, "elbow_flex": 150.0,
        "wrist_flex": 50.0, "wrist_yaw": 50.0, "wrist_roll": 50.0})
    mit_kd: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 3.0, "shoulder_lift": 10.0, "elbow_flex": 10.0,
        "wrist_flex": 5.0, "wrist_yaw": 4.0, "wrist_roll": 4.0})
    gravity_mit_kp: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 50.0, "shoulder_lift": 50.0, "elbow_flex": 50.0,
        "wrist_flex": 30.0, "wrist_yaw": 50.0, "wrist_roll": 50.0})
    gravity_mit_kd: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 3.0, "shoulder_lift": 5.0, "elbow_flex": 5.0,
        "wrist_flex": 3.0, "wrist_yaw": 4.0, "wrist_roll": 4.0})
    motor_can_ids: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "shoulder_pan": (0x01, 0xFD), "shoulder_lift": (0x02, 0xFD), "elbow_flex": (0x03, 0xFD),
        "wrist_flex": (0x04, 0xFD), "wrist_yaw": (0x05, 0xFD), "wrist_roll": (0x06, 0xFD),
        "gripper": (0x07, 0xFD)})
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "shoulder_pan": (-145.0, 145.0), "shoulder_lift": (-0.0, 170.0), "elbow_flex": (-0.0, 200.0),
        "wrist_flex": (-80.0, 90.0), "wrist_yaw": (-90.0, 90.0), "wrist_roll": (-130.0, 130.0),
        "gripper": (-0.0, 270.0)})
    joint_directions: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 1.0, "shoulder_lift": 1.0, "elbow_flex": -1.0, "wrist_flex": -1.0,
        "wrist_yaw": -1.0, "wrist_roll": 1.0, "gripper": 6.0})
    gravity_compensation: bool = False
    # NEW: with gravity_compensation, keep the stiff mit_kp/kd gains (instead of the plugin's softer gravity gains):
    # stiff tracking + no sag. Used by free-mode replay and run_policy so demos and rollouts share one controller.
    gravity_stiff: bool = False
    # NEW: per-joint multipliers on the URDF gravity torque (fitted by `python -m b601free.free`, key a)
    gravity_scales: dict[str, float] = field(default_factory=dict)
    # NEW: fitted link masses/COMs (b601free.free calibration); applied to the URDF gravity model when present
    gravity_fit_path: str | None = str(Path(__file__).resolve().parent / "gravity_fit.json")
    # NEW: the gripper is taped (mallet): never command it. No zero-detect on connect, zero torque every tick
    # (feedback still flows), no safe_zero gripper move; the recorded gripper "action" is its measured position.
    gripper_locked: bool = True
    gravity_urdf_path: str = DEFAULT_URDF
    gripper_mit_kp: float = 12.0
    gripper_mit_kd: float = 0.05
    gripper_mit_torque_limit: float = 3.5
    gripper_mit_hold_torque_limit: float = 1.0
    pos_vel_velocity: list[float] = field(default_factory=lambda: [50, 0.4, 0.4, 50, 50, 50, 0])  # DM only
    temp_alarm_threshold_c: float = 115.0
    temp_overheat_threshold_c: float = 125.0
    temp_emergency_disable_threshold_c: float = 135.0
    motor_model_mapping: dict[str, str] = field(default_factory=lambda: {
        "shoulder_pan": "rs-06", "shoulder_lift": "rs-06", "elbow_flex": "rs-06",
        "wrist_flex": "rs-00", "wrist_yaw": "rs-00", "wrist_roll": "rs-00", "gripper": "rs-00"})


def ensure_safe_goal_position(goal_present_pos: dict[str, tuple[float, float]],
                              max_relative_target: float | dict[str, float]) -> dict[str, float]:
    """lerobot.robots.utils.ensure_safe_goal_position, verbatim (ints accepted too)."""
    if isinstance(max_relative_target, (int, float)):
        diff_cap = dict.fromkeys(goal_present_pos, float(max_relative_target))
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)
    warnings_dict, safe = {}, {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        max_diff = diff_cap[key]
        safe_diff = max(min(goal_pos - present_pos, max_diff), -max_diff)
        safe[key] = present_pos + safe_diff
        if abs(safe[key] - goal_pos) > 1e-4:
            warnings_dict[key] = {"original goal_pos": goal_pos, "safe goal_pos": safe[key]}
    if warnings_dict:
        logger.warning("Relative goal position magnitude had to be clamped to be safe.\n%s",
                       pformat(warnings_dict, indent=4))
    return safe


def action_to_motor(action: dict[str, float] | np.ndarray, cfg: B601Config) -> dict[str, float]:
    """Plugin send_action() mapping: action-frame degrees (what the leader's get_action() returns) ->
    motor-frame degrees: `position * joint_directions[joint]`, then clip to joint_limits, and a missing
    wrist_yaw is commanded to 0 (6-DOF leader support)."""
    if isinstance(action, np.ndarray):
        action = {f"{n}.pos": float(v) for n, v in zip(JOINTS, action)}
    goal = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")}
    for name, position in goal.items():
        position = position * cfg.joint_directions.get(name, 0.0)  # plugin default 0.0 (!)
        if name in cfg.joint_limits:
            lo, hi = cfg.joint_limits[name]
            position = max(lo, min(hi, position))
        goal[name] = position
    if "wrist_yaw" not in goal:
        goal["wrist_yaw"] = 0.0
    return goal


class B601:
    name = "seeed_b601_rs_follower"
    motor_type = "rs"

    def __init__(self, cfg: B601Config, controller_factory: Callable[[B601Config], Any] | None = None):
        self.cfg = self.config = cfg
        self._controller_factory = controller_factory
        self.bus = None
        self.motors: dict[str, Any] = {}
        self.motor_names = list(cfg.motor_can_ids.keys())
        if self.motor_names != JOINTS:
            raise ValueError(f"motor order must be {JOINTS}")
        self._in_safe_zero = False
        self._emergency_disable_requested = False
        self._gravity_pairing: dict[str, int] | None = None
        self._gravity_model = None
        self._skip_safe_zero_on_disconnect = False
        self.calibration_fpath = calibration_path("robots", self.name, cfg.id, cfg.calibration_dir)
        self.calibration = load_calibration(self.calibration_fpath)
        # RS gripper impedance state (plugin lazily creates these with hasattr)
        self._gripper_prev_target_pos = None
        self._gripper_prev_filtered_target_vel = None
        self._gripper_prev_state_pos = None
        self._gripper_open_stall_count = 0
        self._last_cmd: dict[str, float] = {}  # motor-frame degrees of the last command per joint
        self.torque_enabled = False
        self.last_temps: dict[str, float] = {}
        self.missing: list[str] = []
        self._missing_warn_t = 0.0

    def __repr__(self):
        return f"{self.cfg.id} B601RS"

    # ------------------------------------------------------------------ connection
    @property
    def is_connected(self) -> bool:
        return self.bus is not None

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def _open_bus(self):
        if self._controller_factory is not None:
            return self._controller_factory(self.cfg)
        from motorbridge import Controller as MotorBridgeController
        if self.cfg.can_adapter == "damiao":
            return MotorBridgeController.from_dm_serial(serial_port=self.cfg.port, baud=self.cfg.dm_serial_baud)
        if self.cfg.can_adapter == "robstride":
            raise NotImplementedError("RobStride dedicated USB-to-CAN adapter is not supported by motorbridge.")
        return MotorBridgeController(channel=self.cfg.port)

    def _add_motors_to_bus(self):
        for motor_name, (send_id, recv_id) in self.cfg.motor_can_ids.items():
            self.motors[motor_name] = self.bus.add_robstride_motor(
                send_id, recv_id, self.cfg.motor_model_mapping[motor_name])

    def connect(self, calibrate: bool = True, enable_torque: bool = True) -> None:
        """Plugin connect(): open bus, add motors, calibration check, configure (MIT mode, enable),
        gripper zero detection, gravity pairing. enable_torque=False = read-only (torque stays off,
        no gripper probe)."""
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")
        logger.info("Connecting arm on %s (adapter=%s)...", self.cfg.port, self.cfg.can_adapter)
        self.bus = self._open_bus()
        try:
            self._add_motors_to_bus()
            if not self.is_calibrated and calibrate:
                logger.info("No calibration file found at %s", self.calibration_fpath)
                self.calibrate()
                self._skip_safe_zero_on_disconnect = False
            self.configure(enable=enable_torque)
            self._wait_for_feedback(strict=enable_torque)
            if calibrate and enable_torque and not self.cfg.gripper_locked:
                self.detect_gripper_zero()
            self._build_gravity_pairing()
        except BaseException:
            logger.error("connect failed; disabling motors and closing the bus")
            self._close_bus(disable=True)
            raise
        logger.info("%s connected (torque %s).", self, "ON" if enable_torque else "OFF")

    def calibrate(self) -> None:
        """Plugin calibrate(): torque off, user puts the arm in its zero pose (gripper closed), each motor
        stores its zero, then the default calibration json is written."""
        self._skip_safe_zero_on_disconnect = True
        if self.calibration:
            user_input = input(f"Press ENTER to use provided calibration file associated with the id {self.cfg.id}, "
                               "or type 'c' and press ENTER to run calibration: ")
            if user_input.strip().lower() != "c":
                logger.info("Using calibration file associated with the id %s", self.cfg.id)
                return
        logger.info("Running calibration for %s", self)
        self.bus.disable_all()
        print("\nCalibration: Set Zero Position\n"
              "Please MANUALLY move the robot to its ZERO POSITION, and close its gripper.\n"
              "Reference the B601 manual for Zero Pose (generally the default sit-down position).\n")
        input("Press ENTER when ready...")
        for motor in self.motors.values():
            motor.set_zero_position()
            time.sleep(LONG_TIMEOUT_SEC)
        logger.info("Arm zero position set. Setting range: -90 to +90 by default for all joints")
        self.calibration = {name: default_calibration_entry(send_id)
                            for name, (send_id, _) in self.cfg.motor_can_ids.items()}
        save_calibration(self.calibration_fpath, self.calibration)
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self, enable: bool = True) -> None:
        """Torque off while switching modes, MIT mode for every RS motor (10 tries), then enable all."""
        self.bus.disable_all()
        self.torque_enabled = False
        num_retry = 9
        for motor_name, motor in self.motors.items():
            for attempt in range(num_retry + 1):
                try:
                    motor.ensure_mode(MIT_MODE)
                    break
                except Exception:
                    if attempt == num_retry:
                        raise
                    time.sleep(MEDIUM_TIMEOUT_SEC)
            logger.info("%s ensure mode MIT", motor_name)
        if enable:
            self.bus.enable_all()
            self.torque_enabled = True

    def _wait_for_feedback(self, strict: bool, tries: int = 20) -> None:
        missing = list(self.motors)
        for _ in range(tries):
            for m in self.motors.values():
                m.request_feedback()
            try:
                self.bus.poll_feedback_once()
            except Exception:
                pass
            missing = [n for n, m in self.motors.items() if m.get_state() is None]
            if not missing:
                return
            time.sleep(MEDIUM_TIMEOUT_SEC)
        msg = f"no feedback from motors {missing} (CAN ids {[self.cfg.motor_can_ids[n][0] for n in missing]})"
        if strict:
            raise RuntimeError(msg + "; check power, CAN wiring/termination and `ip -details link show can0`")
        logger.warning(msg)

    def disable_torque(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self.bus.disable_all()
        self.torque_enabled = False
        logger.info("%s torque disabled.", self)

    def detect_gripper_zero(self, tolerance_deg: float = 5.0, mit_tau: float = 0.3) -> None:
        """Plugin detect_gripper_zero() (RS branch): close the gripper with a small MIT torque until the
        position is stable, then require |pos| <= tolerance (the gripper was zeroed closed)."""
        motor = self.motors.get(FOLLOWER_GRIPPER_MOTOR)
        if motor is None:
            return
        lo, hi = self.cfg.joint_limits[FOLLOWER_GRIPPER_MOTOR]
        closing_sign = -math.copysign(1.0, hi if abs(hi) > abs(lo) else lo)
        prev, stable = None, 0
        for _ in range(200):
            motor.send_mit(0.0, 0.0, 0.0, 1.5, closing_sign * mit_tau)
            time.sleep(0.05)
            motor.request_feedback()
            try:
                self.bus.poll_feedback_once()
            except Exception:
                pass
            state = motor.get_state()
            if state is None:
                continue
            cur = math.degrees(state.pos)
            stable = stable + 1 if prev is not None and abs(cur - prev) < 0.1 else 0
            prev = cur
            if stable >= 5:
                break
        motor.send_mit(0.0, 0.0, 0.0, 1.5, 0.0)
        if prev is not None and abs(prev) <= tolerance_deg:
            logger.info("gripper zero-detect: zero-point check passed (%.2f deg)", prev)
        else:
            offset = prev if prev is not None else float("nan")
            raise RuntimeError(f"gripper zero-detect: gripper not homed, offset {offset:.2f} deg, please confirm "
                               "gripper is at home or recalibrate")

    # ------------------------------------------------------------------ temperatures
    def _read_motor_temperatures(self) -> dict[str, float]:
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except Exception:
            logger.warning("Temperature check poll feedback failed.")
        temps = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is not None:
                temps[motor_name] = state.t_mos
        self.last_temps = temps
        return temps

    def _check_motor_temperatures(self, alarm_threshold_c: float, overheat_threshold_c: float,
                                  context: str = "") -> dict[str, float]:
        """HIGH TEMP print above alarm; raise (KeyboardInterrupt subclass) above overheat."""
        temperatures = self._read_motor_temperatures()
        label = f" in {context}" if context else ""
        for motor_name, temp_c in temperatures.items():
            if temp_c > alarm_threshold_c:
                print(f"[HIGH TEMP] {motor_name} t_mos={temp_c:.2f}C > {alarm_threshold_c:.2f}C")
            if temp_c > overheat_threshold_c:
                logger.error("Overheat detected%s: %s t_mos=%.2fC > %.2fC.", label, motor_name, temp_c,
                             overheat_threshold_c)
                raise Overheat("Overheat detected")
        return temperatures

    # ------------------------------------------------------------------ gravity
    def _build_gravity_pairing(self) -> None:
        """motor name -> URDF q index (config order without gripper <-> URDF revolute joints, DFS order)."""
        self._gravity_pairing = None
        if not self.cfg.gravity_compensation:
            return
        if not self.cfg.gravity_urdf_path:
            logger.warning("gravity_urdf_path not set, feedforward disabled")
            return
        from .gravity import load_dynamics_model
        model = load_dynamics_model(urdf_path=self.cfg.gravity_urdf_path)
        arm_motors = [n for n in self.motor_names if n != FOLLOWER_GRIPPER_MOTOR]
        arm_qidx = [j.q_idx for j in model.joints if j.jtype in ("revolute", "continuous")]
        if len(arm_motors) != len(arm_qidx):
            logger.warning("gravity feedforward disabled: arm motor count %d != URDF revolute %d",
                           len(arm_motors), len(arm_qidx))
            return
        fit_path = Path(self.cfg.gravity_fit_path) if self.cfg.gravity_fit_path else None
        if fit_path is not None and fit_path.is_file():
            from .gravity import apply_link_params, link_names
            fit = json.loads(fit_path.read_text())
            if fit.get("links") == link_names(model):
                apply_link_params(model, np.array(fit["theta"]))
                logger.info("gravity model: applied the fitted link parameters from %s (%d samples)",
                            fit_path.name, fit.get("samples", 0))
            else:
                logger.warning("ignoring %s: it was fitted for a different URDF", fit_path)
        self._gravity_model = model
        self._gravity_pairing = dict(zip(arm_motors, arm_qidx))
        logger.info("gravity compensation ON (gains: gravity_mit_kp/kd)")

    def _gravity_feedforward(self, scaled: bool = True) -> dict[str, float]:
        pairing = self._gravity_pairing
        if not pairing:
            return {}
        from .gravity import compute_generalized_gravity
        model = self._gravity_model
        q = np.zeros(model.nq)
        for motor_name, qi in pairing.items():
            state = self.motors[motor_name].get_state()
            if state is not None:
                q[qi] = float(state.pos)
        tau = compute_generalized_gravity(model=model, q=q)
        sc = self.cfg.gravity_scales if scaled else {}
        return {name: float(tau[qi]) * float(sc.get(name, 1.0)) for name, qi in pairing.items()}

    # ------------------------------------------------------------------ I/O
    def read(self) -> dict[str, np.ndarray]:
        """Plugin get_observation() (motors part): q [deg], qvel [deg/s], tau [N*m], 7 each, motor frame.
        A motor without state reads 0.0 like in the plugin; its name is put in self.missing."""
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except Exception:
            logger.warning("can bus poll feedback failed.")
        q, qv, tau = np.zeros(7), np.zeros(7), np.zeros(7)
        missing = []
        for i, (motor_name, motor) in enumerate(self.motors.items()):
            state = motor.get_state()
            if state is not None:
                q[i], qv[i], tau[i] = math.degrees(state.pos), math.degrees(state.vel), state.torq
                self.last_temps[motor_name] = state.t_mos
            else:
                missing.append(motor_name)
        self.missing = missing
        if missing and time.time() - self._missing_warn_t > 1.0:
            self._missing_warn_t = time.time()
            logger.warning("no state for %s (reported as 0.0)", missing)
        return {"q": q, "qvel": qv, "tau": tau}

    def present_deg(self) -> dict[str, float]:
        out = {}
        for name, motor in self.motors.items():
            st = motor.get_state()
            out[name] = math.degrees(st.pos) if st is not None else 0.0
        return out

    def action_to_motor(self, action) -> dict[str, float]:
        return action_to_motor(action, self.cfg)

    def _step_cap(self, name: str) -> float | None:
        s = self.cfg.max_step_deg
        if s is None:
            return None
        return s.get(name) if isinstance(s, dict) else float(s)

    def command(self, targets: np.ndarray | dict[str, float], send: bool = True) -> np.ndarray:
        """Plugin send_action() taking MOTOR-FRAME degrees. Steps: temperature check (skipped inside
        safe_zero, like the plugin), joint-limit clip, missing wrist_yaw -> 0, per-step clamp (new),
        max_relative_target (plugin), gravity feed-forward, MIT send. A dict may hold a subset of joints
        (only those are sent, as in the plugin). send=False: compute everything, send nothing (dry run).
        Returns the 7 motor-frame targets now in force (the values to record as `action`)."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        if send and not self._in_safe_zero:
            self._check_motor_temperatures(self.cfg.temp_alarm_threshold_c, self.cfg.temp_overheat_threshold_c,
                                           context="send_action")
        if isinstance(targets, np.ndarray):
            goal = {n: float(v) for n, v in zip(JOINTS, targets)}
        else:
            goal = {k.removesuffix(".pos"): float(v) for k, v in targets.items()}
        if any(not math.isfinite(v) for v in goal.values()):
            raise ValueError(f"non-finite target {goal}")
        for name, pos in goal.items():
            if name in self.cfg.joint_limits:
                lo, hi = self.cfg.joint_limits[name]
                goal[name] = max(lo, min(hi, pos))
        if "wrist_yaw" not in goal:
            goal["wrist_yaw"] = 0.0

        present = None
        if self.cfg.max_step_deg is not None:
            present = self.present_deg()
            for name, pos in goal.items():
                cap = self._step_cap(name)
                if cap is None:
                    continue
                ref = self._last_cmd.get(name, present[name])
                goal[name] = ref + max(-cap, min(cap, pos - ref))
        if self.cfg.max_relative_target is not None:
            present = present or self.present_deg()
            goal = ensure_safe_goal_position({k: (g, present.get(k, g)) for k, g in goal.items()},
                                             self.cfg.max_relative_target)

        if send:
            gravity_tau = self._gravity_feedforward() if self.cfg.gravity_compensation else {}
            for motor_name, position_degrees in goal.items():
                motor = self.motors.get(motor_name)
                if motor is None:
                    continue
                pos_rad = math.radians(position_degrees)
                if motor_name == FOLLOWER_GRIPPER_MOTOR:
                    if self.cfg.gripper_locked:   # taped: passive, never driven
                        motor.send_mit(0.0, 0.0, 0.0, 0.0, 0.0)
                        st = motor.get_state()
                        if st is not None:
                            goal[motor_name] = math.degrees(st.pos)
                        continue
                    tau_ff = self._gripper_mit_torque(motor, pos_rad)
                    motor.send_mit(0, 0, 0, 1.5, 0.0 if tau_ff is None else tau_ff)
                else:
                    gains = (self.cfg.gravity_mit_kp, self.cfg.gravity_mit_kd) \
                        if self.cfg.gravity_compensation and not self.cfg.gravity_stiff \
                        else (self.cfg.mit_kp, self.cfg.mit_kd)
                    kp, kd = gains[0].get(motor_name, 0.0), gains[1].get(motor_name, 0.0)
                    motor.send_mit(pos_rad, 0, kp, kd, gravity_tau.get(motor_name, 0.0))
        self._last_cmd.update(goal)
        if len(self._last_cmd) < 7:
            present = present or self.present_deg()
        return np.array([self._last_cmd.get(n, present[n] if present else 0.0) for n in JOINTS])

    def _gripper_mit_torque(self, motor, pos_target: float) -> float | None:
        """SeeedB601RSFollower.mit_output_torque_limit(), verbatim: software impedance for the gripper,
        sent as pure MIT torque (kp=0, kd=1.5). Units rad, rad/s, N*m. Assumes 50 Hz (0.02 s) like the
        plugin even though we run at 30 Hz."""
        if motor is None:
            return None
        self.bus.poll_feedback_once()
        state = motor.get_state()
        if state is None:
            return None
        control_dt_s = 0.02
        prev_target_pos = self._gripper_prev_target_pos
        target_vel = 0.0 if prev_target_pos is None else (pos_target - prev_target_pos) / control_dt_s
        self._gripper_prev_target_pos = pos_target
        lpf_alpha, target_vel_max = 0.3, 3.0
        prev_filtered_vel = self._gripper_prev_filtered_target_vel
        filtered = target_vel if prev_filtered_vel is None else lpf_alpha * target_vel + (1 - lpf_alpha) * prev_filtered_vel
        target_vel = max(-target_vel_max, min(target_vel_max, filtered))
        self._gripper_prev_filtered_target_vel = target_vel
        prev_state_pos = self._gripper_prev_state_pos
        estimated_state_vel = 0.0 if prev_state_pos is None else (state.pos - prev_state_pos) / control_dt_s
        self._gripper_prev_state_pos = state.pos
        kp, kd = float(self.cfg.gripper_mit_kp), float(self.cfg.gripper_mit_kd)
        impedance_torque = kp * (pos_target - state.pos) + kd * (target_vel - state.vel)
        stalled = abs(estimated_state_vel) < 0.25
        # open-direction stall (RS opens positive): cut torque after 5 stalled cycles
        if (pos_target - state.pos) > 0.05 and stalled:
            self._gripper_open_stall_count += 1
        else:
            self._gripper_open_stall_count = 0
        if self._gripper_open_stall_count >= 5:
            logger.warning("gripper open-stall detected, cutting torque")
            motor.request_feedback()
            return 0.0
        max_torque = self.cfg.gripper_mit_torque_limit if not stalled else self.cfg.gripper_mit_hold_torque_limit
        motor.request_feedback()
        return max(-max_torque, min(max_torque, impedance_torque))

    def hold(self, keep_gripper_target: bool = True, send: bool = True) -> np.ndarray:
        """Command the arm to its measured position (gripper keeps its last target so a grasp is kept)."""
        target = self.read()["q"].copy()
        if keep_gripper_target and GRIPPER in self._last_cmd:
            target[6] = self._last_cmd[GRIPPER]
        return self.command(target, send=send)

    # ------------------------------------------------------------------ safe zero
    def safe_zero(self, step_interval_s: float = 0.02, exit_on_complete: bool = True) -> None:
        """Plugin safe_zero(): two-stage interpolation back to zero. Stage 1: CAN ids 1/4/5/6 -> 0 while
        2/3 hold; stage 2: ids 2/3 -> 0 (and a gripper past 180 deg back to 170). 0.5 deg per frame,
        `step_interval_s` per frame, temperature emergency-disable check on every frame."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        if self._in_safe_zero:
            logger.warning("safe_zero skipped: already running.")
            return
        if step_interval_s < 0.0:
            raise ValueError("step_interval_s must be >= 0")
        self._in_safe_zero = True
        try:
            id_to_joint = {sid: n for n, (sid, _) in self.cfg.motor_can_ids.items() if n != FOLLOWER_GRIPPER_MOTOR}
            stage_1 = [id_to_joint[i] for i in (1, 4, 5, 6) if i in id_to_joint]
            stage_2 = [id_to_joint[i] for i in (2, 3) if i in id_to_joint]
            if not stage_1 + stage_2:
                logger.warning("safe_zero skipped: no arm joints mapped to CAN IDs 1-6.")
                return

            def _read_motor_deg(joint_name: str) -> float:
                motor = self.motors.get(joint_name)
                if motor is None:
                    raise RuntimeError(f"safe_zero failed: motor '{joint_name}' not found")
                max_retry = 10
                for attempt in range(1, max_retry + 1):
                    try:
                        motor.request_feedback()
                        self.bus.poll_feedback_once()
                    except Exception:
                        logger.debug("safe_zero feedback poll failed for %s (%d/%d)", joint_name, attempt, max_retry)
                    state = motor.get_state()
                    if state is not None:
                        return math.degrees(state.pos)
                    if attempt < max_retry:
                        time.sleep(MEDIUM_TIMEOUT_SEC)
                raise RuntimeError(f"safe_zero failed: unable to read state for '{joint_name}' after {max_retry} attempts")

            def _interp_to_zero(active_starts, hold_joints, targets=None) -> bool:
                if not active_starts:
                    return False
                targets = targets or {}
                max_delta = max((abs(targets.get(j, 0.0) - v) for j, v in active_starts.items()), default=0.0)
                frames = max(1, math.ceil(max_delta * 2.0))
                for frame in range(1, frames + 1):
                    for motor_name, temp_c in self._read_motor_temperatures().items():
                        if temp_c > self.cfg.temp_emergency_disable_threshold_c:
                            logger.error("Auto-disable on overtemperature during safe_zero: %s t_mos=%.2fC > %.2fC.",
                                         motor_name, temp_c, self.cfg.temp_emergency_disable_threshold_c)
                            self._emergency_disable_requested = True
                            self.disable_torque()
                            logger.error("safe_zero aborted: emergency overtemperature.")
                            return True
                    ratio = frame / frames
                    goal = dict(hold_joints)
                    for joint, start in active_starts.items():
                        goal[joint] = start + (targets.get(joint, 0.0) - start) * ratio
                    self.command(goal)  # motor frame already: no /direction needed (plugin cancels it)
                    if step_interval_s > 0.0:
                        time.sleep(step_interval_s)
                return False

            stage_1_start = {j: _read_motor_deg(j) for j in stage_1}
            stage_2_start = {j: _read_motor_deg(j) for j in stage_2}
            logger.info("safe_zero stage1 start: joints=%s", stage_1)
            if _interp_to_zero(stage_1_start, stage_2_start):
                return
            stage_2_active, stage_2_targets = dict(stage_2_start), {}
            if FOLLOWER_GRIPPER_MOTOR in self.motors and not self.cfg.gripper_locked:
                try:
                    g = _read_motor_deg(FOLLOWER_GRIPPER_MOTOR)
                except RuntimeError as e:
                    logger.warning("safe_zero: could not read gripper position: %s", e)
                    g = None
                if g is not None and abs(g) > 180.0:
                    stage_2_active[FOLLOWER_GRIPPER_MOTOR] = g
                    stage_2_targets[FOLLOWER_GRIPPER_MOTOR] = math.copysign(170.0, g)
                    logger.info("safe_zero gripper: %.2f deg (motor, abs>180), returning to %.2f", g,
                                stage_2_targets[FOLLOWER_GRIPPER_MOTOR])
            logger.info("safe_zero stage2 start: joints=%s", stage_2)
            if _interp_to_zero(stage_2_active, {j: 0.0 for j in stage_1}, stage_2_targets):
                return
            logger.info("safe_zero done.")
            time.sleep(2.0)
            if exit_on_complete:
                raise KeyboardInterrupt("safe_zero completed")
        finally:
            self._in_safe_zero = False

    # ------------------------------------------------------------------ shutdown
    def _close_bus(self, disable: bool):
        for name, motor in self.motors.items():
            for fn in ((motor.disable,) if disable else ()) + (motor.clear_error, motor.close):
                try:
                    fn()
                except Exception as e:  # keep going: disable every motor we can
                    logger.error("%s.%s failed: %s", name, fn.__name__, e)
        try:
            if self.bus is not None:
                self.bus.close()
        finally:
            self.bus = None
            self.motors = {}
            self.torque_enabled = False

    def disconnect(self, safe_zero: bool = True) -> None:
        """Plugin disconnect(): safe_zero() (unless skipped), then disable + clear_error + close every motor.
        A Ctrl-C during safe_zero skips the rest of the motion and disables torque immediately."""
        if not self.is_connected:
            return
        if safe_zero and self.torque_enabled and not self._in_safe_zero and not self._emergency_disable_requested \
                and not self._skip_safe_zero_on_disconnect:
            print("[b601] returning to zero (safe_zero). Ctrl-C to abort and disable torque NOW "
                  "(the arm will drop: support it).", flush=True)
            try:
                self.safe_zero(exit_on_complete=False)
            except KeyboardInterrupt:
                logger.warning("safe_zero interrupted: disabling torque where the arm is.")
            except Exception:
                logger.exception("safe_zero during disconnect failed.")
        elif self.torque_enabled and not safe_zero:
            logger.warning("disabling torque WITHOUT safe_zero: the arm will drop under gravity unless supported.")
        self._close_bus(disable=self.cfg.disable_torque_on_disconnect)
        self._emergency_disable_requested = False
        logger.info("%s disconnected.", self)


# ---------------------------------------------------------------------- CLI
def make_follower(args) -> B601:
    """Build a B601 from the shared CLI args (see add_follower_args)."""
    cfg = B601Config(port=args.can, id=args.follower_id, gravity_compensation=args.gravity_comp,
                     gravity_stiff=getattr(args, "gravity_stiff", False),
                     gripper_locked=not getattr(args, "unlock_gripper", False))
    sp = getattr(args, "gravity_scales", None)
    if sp and Path(sp).is_file():
        cfg.gravity_scales = {k: float(v) for k, v in json.loads(Path(sp).read_text()).items() if k in JOINTS}
    factory = None
    if args.mock:
        from .mock import FakeController
        factory = FakeController.factory
    arm = B601(cfg, controller_factory=factory)
    if args.mock and not arm.is_calibrated:  # never prompt in mock mode
        arm.calibration = {n: default_calibration_entry(s) for n, (s, _) in cfg.motor_can_ids.items()}
    return arm


def add_follower_args(p: argparse.ArgumentParser):
    p.add_argument("--can", default="can0", help="SocketCAN interface")
    p.add_argument("--follower_id", default="follower1", help="calibration id (lerobot --robot.id)")
    p.add_argument("--gravity_comp", action="store_true", help="plugin gravity feed-forward (+ its softer gains)")
    p.add_argument("--unlock_gripper", action="store_true",
                   help="allow gripper motion (default: LOCKED, the gripper is taped and never commanded)")
    p.add_argument("--gravity_stiff", action="store_true",
                   help="with --gravity_comp: keep the stiff default gains (what b601free.free replay records with)")
    p.add_argument("--gravity_scales", default=str(Path(__file__).resolve().parent / "gravity_scales.json"),
                   help="per-joint gravity scales fitted by b601free.free (used when the file exists)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["read", "calibrate"])
    add_follower_args(p)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)
    arm = make_follower(args)
    if args.cmd == "calibrate":
        arm.bus = arm._open_bus()
        try:
            arm._add_motors_to_bus()
            arm.calibration = {}
            arm.calibrate()
        finally:
            arm._close_bus(disable=True)
        return
    arm.connect(calibrate=False, enable_torque=False)
    print("torque is OFF. Joint order:", " ".join(f"{j[:9]:>9}" for j in JOINTS))
    try:
        while True:
            o = arm.read()
            t = arm.last_temps
            print(f"\rq[deg] {fmt_vec(o['q'])} | tmax {max(t.values()) if t else float('nan'):5.1f}C "
                  f"{'MISSING ' + ','.join(arm.missing) if arm.missing else ''}   ", end="", flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print()
    finally:
        arm.disconnect(safe_zero=False)


if __name__ == "__main__":
    main()
