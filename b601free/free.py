"""Free-arm (kinesthetic) mode for the B601-RS follower: gravity + friction compensated, zero stiffness, so you can
move the arm by hand. Records kinesthetic demos and REPLAYS them to collect clean training data (no hand in view).

    python -m b601free.free --skip_gripper_check                                   # free mode + calibration only
    python -m b601free.free --skip_gripper_check --out data/xylo --ready 1 \\
        --instructions "hit the red bar|hit the blue bar" --front <overhead> --wrist <wrist>   # demo collection

Starts in HOLD (soft position hold + gravity feed-forward). Keys:
  f        HOLD -> FREE (stiffness ramps to 0 over --ramp_s while gravity torque ramps in: SUPPORT THE ARM)
           FREE -> HOLD at the current pose (instant, no jump)
  a        AUTO-calibrate here: the arm moves itself through 9 poses (+-25 deg lift/elbow/wrist), measures holding
           torque (gravity) and moving torque (friction), comes back, fits + saves. Start up in free space.
  h / z    go to the --ready pose (h) / home (z: pan+wrist first, then lift+elbow, the safe_zero order), then HOLD
  r        (FREE) start recording a kinesthetic demo        s  stop + save (replay first, unless --no_replay)
  d        discard the recording                            i  next instruction (--instructions "a|b|c")
  0-9      save the current pose to slot N in --poses       p  print the current pose
  c / x    manual calibration sample here / fit           j + -  select a joint / nudge its gravity scale
  o / g    gripper open / close (holds an object with the plugin's torque limit)
  q        quit: hold, slow return to zero (safe_zero), torque off. Ctrl-C = same; 2nd Ctrl-C drops the arm.
Replay (after s): the arm moves to where the demo started, waits --wait_s (STEP BACK), then replays your motion with
the stiff controller run_policy uses with --gravity_comp --gravity_stiff, while recording cameras at --fps:
action = your trajectory --lead_ms ahead, q/qvel/tau = what the arm did. Any key during replay aborts (discarded).
Safety: joint-limit springs, FREE -> HOLD above --max_qvel, ramp drift abort, blocked-joint aborts in AUTO/replay,
motor temperature checks every 0.3 s (plugin thresholds).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .b601 import FOLLOWER_GRIPPER_MOTOR, add_follower_args, make_follower
from .cameras import add_camera_args, make_rig
from .common import ARM_JOINTS, JOINTS, PKG, RateLoop, fmt_vec, setup_logging
from .keys import KeyReader

# AUTO calibration poses: (lift, elbow, wrist_flex) offsets from the start pose [deg]
AUTO_DELTAS = [(0, 0, 0), (20, 0, 0), (-20, 0, 0), (0, 25, 0), (0, -25, 0), (0, 0, 30), (0, 0, -30),
               (15, -20, 20), (-15, 20, -20)]
SHORT = ["pan", "lift", "elbow", "wflex", "wyaw", "roll"]
FREE_KD = {"shoulder_pan": 0.5, "shoulder_lift": 1.0, "elbow_flex": 1.0,
           "wrist_flex": 0.3, "wrist_yaw": 0.3, "wrist_roll": 0.3}  # N*m*s/rad: light damping only


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_follower_args(p)
    add_camera_args(p)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--skip_gripper_check", action="store_true", help="gripper holds a tool (mallet)")
    p.add_argument("--hz", type=float, default=120.0, help="control loop rate (a multiple of --fps)")
    p.add_argument("--ramp_s", type=float, default=3.0, help="HOLD -> FREE stiffness ramp time")
    p.add_argument("--kd_scale", type=float, default=1.0, help="scale the FREE damping")
    p.add_argument("--fric_scale", type=float, default=0.7, help="apply this fraction of the measured friction")
    p.add_argument("--fric_dead", type=float, default=2.0, help="no friction assist below this speed [deg/s]")
    p.add_argument("--max_qvel", type=float, default=150.0, help="FREE -> HOLD above this pan/lift/elbow speed [deg/s]")
    p.add_argument("--max_qvel_wrist", type=float, default=400.0, help="same for the light wrist joints (strikes)")
    p.add_argument("--max_drift", type=float, default=15.0, help="abort the ramp if a joint drifts this far [deg]")
    p.add_argument("--limit_margin", type=float, default=5.0, help="joint-limit spring starts this far inside [deg]")
    p.add_argument("--limit_kp", type=float, default=15.0, help="joint-limit spring stiffness [N*m/rad]")
    p.add_argument("--auto_speed", type=float, default=15.0, help="AUTO calibration joint speed [deg/s]")
    p.add_argument("--goto_speed", type=float, default=30.0, help="ready/home/replay-start moves [deg/s]")
    p.add_argument("--auto_track_err", type=float, default=12.0, help="abort a move if a joint lags this far [deg]")
    p.add_argument("--poses", default=str(PKG / "poses.json"))
    p.add_argument("--samples_file", default=str(PKG / "gravity_samples.json"),
                   help="calibration samples, kept across runs (delete to start over)")
    p.add_argument("--log_dir", default=str(PKG / "logs"), help="30 Hz CSV log of every session")
    p.add_argument("--ready", default=None, help="pose slot for 'h' (demo start/end pose)")
    p.add_argument("--grip_open", type=float, default=90.0, help="'o' opens the gripper to this [motor deg]")
    # recording
    p.add_argument("--out", default=None, help="dataset dir (enables r/s/d recording; needs cameras)")
    p.add_argument("--instructions", default=None, help="'|'-separated instructions, 'i' cycles")
    p.add_argument("--instruction", default="play the xylophone")
    p.add_argument("--fps", type=float, default=30.0, help="recorded frame rate (= run_policy rate)")
    p.add_argument("--no_replay", action="store_true", help="save the kinesthetic pass itself (hand in view)")
    p.add_argument("--lead_ms", type=float, default=100.0, help="action = demo trajectory this far ahead")
    p.add_argument("--wait_s", type=float, default=2.0, help="pause before replay (step back)")
    p.add_argument("--replay_max_step", type=float, default=8.0, help="replay per-frame step cap [deg] (run_policy "
                                                                       "--max_step must allow the same)")
    p.add_argument("--replay_err", type=float, default=20.0, help="abort replay if a joint is this far off [deg]")
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--max_ep_s", type=float, default=60.0)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    a.instr_list = [s.strip() for s in a.instructions.split("|")] if a.instructions else [a.instruction]
    return a


def interp(t_src, x_src, t):
    """Linear interpolation of (T, D) samples at times t (clamped at the ends)."""
    return np.stack([np.interp(t, t_src, x_src[:, d]) for d in range(x_src.shape[1])], -1)


class Free:
    def __init__(self, arm, args, rig=None, saver=None):
        self.arm, self.args, self.rig, self.saver = arm, args, rig, saver
        self.mode = "HOLD"            # HOLD | RAMP | FREE | SAMPLE | AUTO | REPLAY
        self.auto = None
        self.hold_q = None            # rad, arm joints
        self.ramp_t0 = 0.0
        self.sample = None
        self.samples = []             # [(q[6] rad, tau_meas[6] N*m)], persisted in --samples_file
        sf = Path(args.samples_file)
        if sf.is_file():
            self.samples = [(np.array(d["q"]), np.array(d["tau"])) for d in json.loads(sf.read_text())]
            print(f"loaded {len(self.samples)} calibration samples from {sf.name}")
        self.log = None
        self.after_sample = "HOLD"
        self.fast_ticks = 0
        self.scales = {j: 1.0 for j in ARM_JOINTS}
        self.friction = {j: 0.0 for j in ARM_JOINTS}   # Coulomb friction [N*m], measured by AUTO calibration
        sp = Path(args.gravity_scales)
        if sp.is_file():
            saved = json.loads(sp.read_text())
            self.friction.update(saved.pop("friction", {}))
            self.scales.update({k: v for k, v in saved.items() if k in self.scales})
            print(f"loaded gravity scales {self.scales}\n       friction {self.friction} from {sp}")
        self.arm.cfg.gravity_scales = dict(self.scales)
        self.qd_f = np.zeros(6)
        self.fric_samples = {j: [] for j in ARM_JOINTS}
        self.sel = 1
        self.grip_target = None       # rad
        self.instr_i = 0
        self.rec = None               # kinesthetic recording
        self.rp = None                # replay state
        self.msg = ""

    # ---------------------------------------------------------------- state
    def states(self):
        st = {n: m.get_state() for n, m in self.arm.motors.items()}
        missing = [n for n in JOINTS if st.get(n) is None]
        if missing:
            raise RuntimeError(f"no feedback from {missing}")
        q = np.array([st[n].pos for n in JOINTS])
        qd = np.array([st[n].vel for n in JOINTS])
        tau = np.array([st[n].torq for n in JOINTS])
        return q, qd, tau

    def gravity(self):
        g = self.arm._gravity_feedforward(scaled=False)
        return np.array([g.get(n, 0.0) for n in ARM_JOINTS])

    def enter_hold(self, q, why=""):
        self.mode, self.hold_q = "HOLD", q.copy()
        if why:
            self.msg = why

    @property
    def instruction(self):
        return self.args.instr_list[self.instr_i]

    # ---------------------------------------------------------------- one tick
    def tick(self, now, tick):
        arm, a = self.arm, self.args
        arm.bus.poll_feedback_once()
        q7, qd7, tau7 = self.states()
        q, qd, tau_meas = q7[:6], qd7[:6], tau7[:6]
        if self.grip_target is None:
            self.grip_target = q7[6]
        if self.hold_q is None:
            self.hold_q = q.copy()
        if self.mode == "REPLAY":
            self._replay_step(now, tick, q7, qd7, tau7)
            return q, qd, np.zeros(6)
        if self.rec is not None:
            self._record_step(now, tick, q7, qd7, tau7)
        g = self.gravity()
        s = np.array([self.scales[j] for j in ARM_JOINTS])
        ff = s * g
        self.qd_f = 0.6 * self.qd_f + 0.4 * qd
        if self.mode == "FREE" and a.fric_scale > 0:  # push along the motion: cancels most of the gearbox friction
            v = np.degrees(self.qd_f)
            ramp = np.clip((np.abs(v) - a.fric_dead) / 4.0, 0.0, 1.0)
            fc = np.array([self.friction[j] for j in ARM_JOINTS])
            ff = ff + a.fric_scale * fc * np.sign(v) * ramp
        hold_kp = np.array([arm.cfg.gravity_mit_kp[j] for j in ARM_JOINTS])
        hold_kd = np.array([arm.cfg.gravity_mit_kd[j] for j in ARM_JOINTS])
        free_kd = np.array([FREE_KD[j] for j in ARM_JOINTS]) * a.kd_scale

        if self.mode == "RAMP":
            r = min(1.0, (now - self.ramp_t0) / a.ramp_s)
            drift = np.degrees(np.abs(q - self.hold_q))
            if drift.max() > a.max_drift:
                j = ARM_JOINTS[int(drift.argmax())]
                self.enter_hold(q, f"ramp aborted: {j} drifted {drift.max():.0f} deg (gravity model/scale off?)")
            elif r >= 1.0:
                self.mode, self.msg = "FREE", ("FREE: r = record a demo" if self.saver else "FREE: move it by hand")
        if self.mode == "FREE":
            lim = np.array([a.max_qvel] * 3 + [a.max_qvel_wrist] * 3)
            fast = (np.degrees(np.abs(qd)) > lim).any()
            self.fast_ticks = self.fast_ticks + 1 if fast else 0
            if self.fast_ticks >= 3:
                self.enter_hold(q, f"speed watchdog: {np.degrees(np.abs(qd)).max():.0f} deg/s -> HOLD")
        if self.mode == "SAMPLE":
            if self._sample_step(now, q, qd, tau_meas, g) is not None:
                self._leave_sample(now)
        if self.mode == "AUTO":
            self._auto_step(now, q, qd, tau_meas, g)
            if self.mode == "REPLAY":   # a replay's move-to-start just finished
                return q, qd, ff

        if self.mode in ("HOLD", "SAMPLE", "AUTO"):
            pos, kp, kd = self.hold_q, hold_kp, hold_kd
        elif self.mode == "RAMP":
            r = min(1.0, (now - self.ramp_t0) / a.ramp_s)
            pos, kp, kd = self.hold_q, (1 - r) * hold_kp, free_kd + (1 - r) * (hold_kd - free_kd)
        else:  # FREE
            pos, kp, kd = q.copy(), np.zeros(6), free_kd
        pos, kp = pos.copy(), kp.copy()
        if self.mode in ("RAMP", "FREE"):  # joint-limit springs
            m = math.radians(a.limit_margin)
            for i, j in enumerate(ARM_JOINTS):
                lo, hi = (math.radians(v) for v in arm.cfg.joint_limits[j])
                home_stop = j in ("shoulder_lift", "elbow_flex") and lo == 0.0  # rest pose sits on these stops
                if q[i] < lo + m and not home_stop:
                    pos[i], kp[i] = lo + m, max(kp[i], a.limit_kp)
                elif q[i] > hi - m:
                    pos[i], kp[i] = hi - m, max(kp[i], a.limit_kp)
        for i, j in enumerate(ARM_JOINTS):
            arm.motors[j].send_mit(float(pos[i]), 0.0, float(kp[i]), float(kd[i]), float(ff[i]))
        if self.frame_tick(tick):
            self.log_row(now, q, qd, tau_meas, ff)
        gm = arm.motors[FOLLOWER_GRIPPER_MOTOR]
        if arm.cfg.gripper_locked:   # taped: passive, never driven
            gm.send_mit(0.0, 0.0, 0.0, 0.0, 0.0)
            self.grip_target = q7[6]
        else:
            gt = arm._gripper_mit_torque(gm, self.grip_target)
            gm.send_mit(0.0, 0.0, 0.0, 1.5, 0.0 if gt is None else gt)
        arm._last_cmd.update({j: math.degrees(v) for j, v in zip(ARM_JOINTS, q)})
        arm._last_cmd[FOLLOWER_GRIPPER_MOTOR] = math.degrees(self.grip_target)
        return q, qd, ff

    # ---------------------------------------------------------------- recording (kinesthetic pass)
    def frame_tick(self, tick):
        return tick % max(1, int(round(self.args.hz / self.args.fps))) == 0

    def start_rec(self, now):
        self.rec = {"t0": now, "t": [], "q": [], "frames": []}
        self.msg = f"REC: '{self.instruction}'  s = save, d = discard"

    def _record_step(self, now, tick, q7, qd7, tau7):
        r = self.rec
        r["t"].append(now - r["t0"])
        r["q"].append(np.degrees(q7))
        if self.args.no_replay and self.frame_tick(tick):
            f, w, _ = self.rig.read()
            r["frames"].append((now - r["t0"], f, w, np.degrees(q7), np.degrees(qd7), tau7.copy()))
        if now - r["t0"] > self.args.max_ep_s:
            self.msg = f"max {self.args.max_ep_s:.0f}s reached: s = save, d = discard"

    def stop_rec(self, now, q):
        r, self.rec = self.rec, None
        if r is None or len(r["t"]) < 30:
            self.msg = "recording too short: discarded"
            return
        t, qs = np.array(r["t"]), np.array(r["q"])
        if self.args.no_replay:
            ft = np.array([x[0] for x in r["frames"]])
            ep = {"front": [x[1] for x in r["frames"]], "wrist": [x[2] for x in r["frames"]],
                  "q": [x[3] for x in r["frames"]], "qvel": [x[4] for x in r["frames"]],
                  "tau": [x[5] for x in r["frames"]],
                  "action": list(interp(t, qs, ft + self.args.lead_ms / 1000.0))}
            self._save(ep, {"mode": "kinesthetic", "seconds": round(float(t[-1]), 2)})
            self.enter_hold(q, self.msg)
            return
        # replay: move to where the demo started, wait, play it back with the stiff controller
        self.rp = {"t": t, "q": qs, "phase": "goto"}
        self.hold_q, self.mode = q.copy(), "AUTO"
        self.auto = {"wps": [np.radians(qs[0, :6])], "i": 0, "q0": None, "kind": "goto"}
        self._auto_move(now, q.copy(), self.auto["wps"][0], self.args.goto_speed)
        self.msg = f"saved demo ({t[-1]:.1f}s): moving to its start pose for replay"

    def _begin_replay(self, now):
        a, arm = self.args, self.arm
        self.mode = "REPLAY"
        self.rp.update(phase="wait", t_wait=now, ep={k: [] for k in ("front", "wrist", "q", "qvel", "tau", "action")},
                       err=[], bad=0)
        self.msg = f"STEP BACK: replay in {a.wait_s:.0f}s (any key aborts)"
        # the controller run_policy uses with --gravity_comp --gravity_stiff
        arm.cfg.gravity_stiff = True
        arm.cfg.max_step_deg = {j: a.replay_max_step for j in ARM_JOINTS} | {FOLLOWER_GRIPPER_MOTOR: 60.0}
        arm._last_cmd.update({j: float(v) for j, v in zip(JOINTS, self.rp["q"][0])})

    def _replay_step(self, now, tick, q7, qd7, tau7):
        rp, a, arm = self.rp, self.args, self.arm
        if not self.frame_tick(tick):
            return
        if rp["phase"] == "wait":
            arm.command(rp["q"][0])
            if now - rp["t_wait"] >= a.wait_s:
                rp.update(phase="play", t_play=now)
                self.msg = "REPLAYING (any key aborts)"
            return
        tr = now - rp["t_play"]
        target = interp(rp["t"], rp["q"], np.array([tr + a.lead_ms / 1000.0]))[0]
        f, w, _ = self.rig.read()
        sent = arm.command(target)
        ep = rp["ep"]
        for k, v in (("front", f), ("wrist", w), ("q", np.degrees(q7)), ("qvel", np.degrees(qd7)), ("tau", tau7),
                     ("action", sent)):
            ep[k].append(np.asarray(v).copy())
        demo_now = interp(rp["t"], rp["q"], np.array([tr]))[0]
        err = np.abs(np.degrees(q7) - demo_now)[:6]
        rp["err"].append(err)
        rp["bad"] = rp["bad"] + 1 if err.max() > a.replay_err else 0
        if rp["bad"] >= 5:
            self.end_replay(q7[:6], f"replay ABORTED: {ARM_JOINTS[int(err.argmax())]} {err.max():.0f} deg off "
                                    f"(blocked?) - discarded")
            return
        if tr >= rp["t"][-1]:
            e = np.array(rp["err"])
            rms, mx = np.sqrt((e ** 2).mean(0)), e.max(0)
            print(f"\n[replay] tracking vs your demo: rms {fmt_vec(rms, 1)} | max {fmt_vec(mx, 1)} deg "
                  f"(lead {a.lead_ms:.0f} ms)", flush=True)
            self._save(ep, {"mode": "replay", "seconds": round(float(rp["t"][-1]), 2), "lead_ms": a.lead_ms,
                            "track_rms_deg": [round(float(x), 2) for x in rms],
                            "track_max_deg": [round(float(x), 2) for x in mx]})
            self.end_replay(q7[:6], self.msg)

    def end_replay(self, q, why):
        self.arm.cfg.gravity_stiff = False
        self.arm.cfg.max_step_deg = None
        self.rp = None
        self.grip_target = math.radians(self.arm._last_cmd.get(FOLLOWER_GRIPPER_MOTOR, math.degrees(self.grip_target)))
        self.enter_hold(q, why)

    def _save(self, ep, extra):
        n = len(ep["q"])
        info = {"frames": n, "instruction": self.instruction, "time": datetime.now().isoformat(timespec="seconds"),
                "fps": self.args.fps} | extra
        path = self.saver.submit(dict(ep, instruction=self.instruction), info)
        self.msg = f"saved {path.name} ({n} frames). f = free for the next demo"

    # ---------------------------------------------------------------- calibration samples
    def _sample_step(self, now, q, qd, tau_meas, g):
        sm = self.sample
        if sm["phase"] == "settle":
            # still = every joint moved < 0.4 deg over the last 0.4 s (position-based: reported velocity is noisy)
            hist = sm.setdefault("hist", [])
            hist.append((now, np.degrees(q)))
            while hist and now - hist[0][0] > 0.4:
                hist.pop(0)
            span = np.ptp(np.array([h[1] for h in hist]), axis=0) if len(hist) > 1 else np.full(6, np.inf)
            sm["qd_max"] = max(sm.get("qd_max", 0.0), float(np.degrees(np.abs(qd)).max()))
            if now - sm["t_start"] > 0.8 and span.max() < 0.4:
                sm.update(phase="avg", t0=now, tau=[], g=[], note="")
            elif now - sm["t_start"] > 3.0:  # never settled: sample anyway, report how much it moved
                j = SHORT[int(np.argmax(span))]
                sm.update(phase="avg", t0=now, tau=[], g=[],
                          note=f" (not settled: {j} moving {span.max():.1f} deg/0.4s, |qd| up to {sm['qd_max']:.0f} deg/s)")
        elif sm["phase"] == "avg":
            sm["tau"].append(tau_meas)
            sm["g"].append(g)
            sm.setdefault("q", []).append(q.copy())
            if now - sm["t0"] > 0.5:
                gm, tm = np.mean(sm["g"], 0), np.mean(sm["tau"], 0)
                self.samples.append((np.mean(sm["q"], 0), tm))
                self.save_samples()
                ratio = " ".join(f"{j}={t / gm_ if abs(gm_) > 0.2 else float('nan'):4.2f}"
                                 for j, gm_, t in zip(SHORT, gm, tm))
                self.msg = f"sample {len(self.samples)}: measured/model {ratio}"
                print(f"\n[sample {len(self.samples)}] model {fmt_vec(gm, 2)} | measured {fmt_vec(tm, 2)} N*m"
                      f"{sm.get('note', '')}", flush=True)
                self.sample = None
                return "done"
        return None

    def _leave_sample(self, now):
        if self.after_sample == "FREE":
            self.mode, self.ramp_t0 = "RAMP", now
        else:
            self.mode = "HOLD"

    # ---------------------------------------------------------------- automatic moves
    def start_auto(self, q, now):
        lim = self.arm.cfg.joint_limits
        q0, wps = q.copy(), []
        for d in AUTO_DELTAS:
            t = q0.copy()
            for idx, dv in zip((1, 2, 3), d):
                lo, hi = lim[ARM_JOINTS[idx]]
                t[idx] = float(np.clip(q0[idx] + math.radians(dv), math.radians(lo + 10), math.radians(hi - 10)))
            wps.append(t)
        wps.append(q0.copy())  # come back (no sample)
        self.auto = {"wps": wps, "i": 0, "q0": q0, "kind": "cal"}
        self.fric_samples = {j: [] for j in ARM_JOINTS}
        self.hold_q, self.mode = q0.copy(), "AUTO"
        self._auto_move(now, q0, wps[0], self.args.auto_speed)
        self.msg = "AUTO-calibrating: arm moves by itself (any key aborts)"

    def start_home(self, q, now):
        s1 = q.copy()
        s1[[0, 3, 4, 5]] = 0.0          # pan, wrist flex/yaw/roll first
        self.auto = {"wps": [s1, np.zeros(6)], "i": 0, "q0": np.zeros(6), "kind": "home"}
        self.hold_q, self.mode = q.copy(), "AUTO"
        self._auto_move(now, q.copy(), s1, self.args.goto_speed)
        self.msg = "going home (any key aborts)"

    def start_ready(self, q, now):
        poses = json.loads(Path(self.args.poses).read_text()) if Path(self.args.poses).is_file() else {}
        if self.args.ready not in poses:
            self.msg = f"no pose slot {self.args.ready!r} in {Path(self.args.poses).name}: save one with a digit key"
            return
        tgt = np.radians(np.array(poses[self.args.ready]["q_deg"][:6]))
        self.auto = {"wps": [tgt], "i": 0, "q0": tgt, "kind": "ready"}
        self.hold_q, self.mode = q.copy(), "AUTO"
        self._auto_move(now, q.copy(), tgt, self.args.goto_speed)
        self.msg = f"going to ready pose {self.args.ready} (any key aborts)"

    def _auto_move(self, now, frm, to, speed):
        T = max(1.0, float(np.degrees(np.abs(to - frm)).max()) / speed)
        self.auto.update(phase="move", frm=frm.copy(), to=to.copy(), t0=now, T=T, speed=speed)

    def _auto_step(self, now, q, qd, tau_meas, g):
        au, a = self.auto, self.args
        if au["phase"] == "move":
            u = min(1.0, (now - au["t0"]) / au["T"])
            sm = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            self.hold_q = au["frm"] + sm * (au["to"] - au["frm"])
            err = np.degrees(np.abs(q - self.hold_q))
            if au["kind"] == "cal":  # moving at speed: measured - gravity = friction (+ small inertia)
                gtrue = np.array([self.scales[j] for j in ARM_JOINTS]) * g
                for i, j in enumerate(ARM_JOINTS):
                    if abs(math.degrees(qd[i])) > 5.0:
                        self.fric_samples[j].append(math.copysign(1.0, qd[i]) * (tau_meas[i] - gtrue[i]))
            if err.max() > a.auto_track_err:
                j = ARM_JOINTS[int(err.argmax())]
                self.auto, self.rp = None, None
                self.enter_hold(q, f"move aborted: {j} {err.max():.0f} deg off target (blocked?)")
                return
            if u < 1.0:
                return
            last = au["i"] == len(au["wps"]) - 1
            if au["kind"] in ("home", "ready", "goto"):
                if not last:
                    au["i"] += 1
                    self._auto_move(now, self.hold_q.copy(), au["wps"][au["i"]], au["speed"])
                    return
                self.auto = None
                if au["kind"] == "goto":
                    self._begin_replay(now)
                else:
                    self.enter_hold(au["wps"][-1], "home. q = quit" if au["kind"] == "home" else
                                    "at ready pose. f = free, then r = record")
                return
            if last:  # calibration: back at the start pose
                self.auto = None
                self.fit()
                self.fit_friction()
                self.enter_hold(au["q0"], "calibrated + saved. f = free")
                return
            au["phase"] = "sample"
            self.sample = {"phase": "settle", "t_start": now}
        elif au["phase"] == "sample":
            if self._sample_step(now, q, qd, tau_meas, g) is not None:
                au["i"] += 1
                self.msg = f"AUTO: pose {au['i']}/{len(au['wps']) - 1} sampled"
                self._auto_move(now, self.hold_q.copy(), au["wps"][au["i"]], au["speed"])

    # ---------------------------------------------------------------- keys
    def key(self, k, now, q):
        a = self.args
        if self.mode == "REPLAY":
            self.end_replay(q, "replay aborted (key) - discarded")
            return
        if self.mode == "AUTO":
            self.auto, self.sample, self.rp = None, None, None
            self.enter_hold(q, "move aborted (key)")
            return
        if self.rec is not None and k not in ("s", "d"):
            if k == "f":
                self.msg = "recording: s = save, d = discard first"
                return
        if k == "a" and self.mode in ("HOLD", "FREE", "RAMP") and self.rec is None:
            self.start_auto(q, now)
        elif k == "h" and self.mode in ("HOLD", "FREE", "RAMP") and self.rec is None:
            self.start_ready(q, now) if a.ready is not None else self.start_home(q, now)
        elif k == "z" and self.mode in ("HOLD", "FREE", "RAMP") and self.rec is None:
            self.start_home(q, now)
        elif k == "r":
            if self.saver is None:
                self.msg = "recording needs --out and cameras"
            elif self.mode != "FREE":
                self.msg = "press f (FREE) first, then r"
            elif self.rec is None:
                self.start_rec(now)
        elif k == "s" and self.rec is not None:
            self.stop_rec(now, q)
        elif k == "d" and self.rec is not None:
            self.rec, self.msg = None, "recording discarded"
        elif k == "i":
            self.instr_i = (self.instr_i + 1) % len(a.instr_list)
            self.msg = f"instruction: '{self.instruction}'"
        elif k == "f":
            if self.mode == "HOLD":
                self.hold_q, self.mode, self.ramp_t0 = q.copy(), "RAMP", now
                self.msg = f"ramping to FREE over {a.ramp_s:.0f}s: support the arm"
            elif self.mode in ("FREE", "RAMP"):
                self.enter_hold(q, "HOLD")
        elif k == "c" and self.mode != "SAMPLE":
            self.after_sample = "FREE" if self.mode in ("FREE", "RAMP") else "HOLD"
            self.hold_q, self.mode = q.copy(), "SAMPLE"
            self.sample = {"phase": "settle", "t_start": now}
            self.msg = "sampling holding torque: let go of the arm"
        elif k == "x":
            self.fit()
        elif k == "j":
            self.sel = (self.sel + 1) % 6
            self.msg = f"selected {ARM_JOINTS[self.sel]} (scale {self.scales[ARM_JOINTS[self.sel]]:.2f})"
        elif k in ("+", "=", "-", "_"):
            j = ARM_JOINTS[self.sel]
            self.scales[j] = round(min(2.0, max(0.0, self.scales[j] + (0.05 if k in "+=" else -0.05))), 3)
            self.save_scales()
            self.msg = f"{j} gravity scale -> {self.scales[j]:.2f}"
        elif k.isdigit():
            self.save_pose(k, q)
        elif k == "p":
            print(f"\npose (deg): {fmt_vec(np.degrees(q))}", flush=True)
        elif k in ("o", "g") and self.arm.cfg.gripper_locked:
            self.msg = "gripper is LOCKED (taped): not moving it (--unlock_gripper to allow)"
        elif k == "o":
            self.grip_target = math.radians(a.grip_open)
            self.msg = "gripper open"
        elif k == "g":
            self.grip_target = 0.0
            self.msg = "gripper close"

    # ---------------------------------------------------------------- fits + files
    def fit(self):
        """Physical gravity fit: link masses + mass*COM (incl. the camera/mallet payload on the tool link) by
        ridge least squares on every saved sample, pulled toward the URDF values. Replaces the per-joint scales."""
        from .gravity import (_parse_urdf, apply_link_params, gravity_regressor, link_names, link_params)
        if len(self.samples) < 3:
            self.msg = f"need >= 3 samples (have {len(self.samples)}): press a (or c at varied poses)"
            return
        arm = self.arm
        model, pairing = arm._gravity_model, arm._gravity_pairing
        names = link_names(model)
        th0 = link_params(_parse_urdf(arm.cfg.gravity_urdf_path))          # pristine URDF values (the prior)
        sig = []
        for i, n in enumerate(names):
            m0, h0 = th0[4 * i], np.abs(th0[4 * i + 1:4 * i + 4])
            sig += [0.4, 0.04, 0.04, 0.04] if n == "gripper_end" else [0.25 * m0 + 0.02, *(0.25 * h0 + 0.01 * m0 + 0.002)]
        sig, s_tau = np.array(sig), 0.12
        rows = [pairing[j] for j in ARM_JOINTS]
        Ys, Ts = [], []
        for qa, ta in self.samples:
            qm = np.zeros(model.nq)
            qm[rows] = qa
            Ys.append(gravity_regressor(model, qm)[rows])
            Ts.append(ta)
        Y, T = np.concatenate(Ys), np.concatenate(Ts)
        A = np.vstack([Y / s_tau, np.diag(1 / sig)])
        b = np.r_[(T - Y @ th0) / s_tau, np.zeros(len(sig))]
        th = th0 + np.linalg.lstsq(A, b, rcond=None)[0]
        e_urdf = np.abs(T - Y @ th0).reshape(-1, 6).mean(0)
        e_fit = np.abs(T - Y @ th).reshape(-1, 6).mean(0)
        apply_link_params(model, th)
        Path(arm.cfg.gravity_fit_path).write_text(json.dumps(
            {"links": names, "theta": [float(x) for x in th], "samples": len(self.samples),
             "mean_abs_err_urdf": e_urdf.round(3).tolist(), "mean_abs_err_fit": e_fit.round(3).tolist()}, indent=1))
        self.scales = {j: 1.0 for j in ARM_JOINTS}   # the model itself now matches; scales stay for manual nudges
        self.save_scales()
        dm = [f"{n} {th0[4 * i]:.3f}->{th[4 * i]:.3f}kg" for i, n in enumerate(names) if abs(th[4 * i] - th0[4 * i]) > 0.005]
        print(f"\n[fit] physical gravity fit on {len(self.samples)} samples: mean |err| per joint "
              f"URDF {fmt_vec(e_urdf, 2)} -> fit {fmt_vec(e_fit, 2)} N*m\n[fit] mass changes: {', '.join(dm) or 'none'}",
              flush=True)
        self.msg = f"gravity model fitted on {len(self.samples)} samples + saved"

    def fit_friction(self):
        lines = []
        for j in ARM_JOINTS:
            xs = self.fric_samples[j]
            if len(xs) < 20:
                continue
            f = float(np.clip(np.median(xs), 0.0, 1.5))
            self.friction[j] = round(f, 3)
            lines.append(f"{j}: friction {f:.2f} N*m ({len(xs)} moving samples)")
        if lines:
            self.save_scales()
            print("\n[friction] " + "\n[friction] ".join(lines), flush=True)

    def save_samples(self):
        Path(self.args.samples_file).write_text(json.dumps(
            [{"q": [float(x) for x in q], "tau": [float(x) for x in t]} for q, t in self.samples]))

    def open_log(self):
        d = Path(self.args.log_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"free_{datetime.now():%Y%m%d-%H%M%S}.csv"
        self.log = open(path, "w")
        cols = ["t", "mode", "rec"] + [f"{k}_{j}" for k in ("q", "qd", "tau", "ff") for j in SHORT]
        self.log.write(",".join(cols) + "\n")
        return path

    def log_row(self, now, q, qd, tau, ff):
        if self.log is None:
            return
        vals = [f"{now:.3f}", self.mode, "1" if self.rec else "0"]
        vals += [f"{x:.2f}" for x in np.degrees(q)] + [f"{x:.1f}" for x in np.degrees(qd)]
        vals += [f"{x:.3f}" for x in tau] + [f"{x:.3f}" for x in ff]
        self.log.write(",".join(vals) + "\n")

    def save_scales(self):
        self.arm.cfg.gravity_scales = dict(self.scales)
        Path(self.args.gravity_scales).write_text(json.dumps({**self.scales, "friction": self.friction}, indent=1))

    def save_pose(self, slot, q):
        path = Path(self.args.poses)
        poses = json.loads(path.read_text()) if path.is_file() else {}
        grip = math.degrees(self.grip_target) if self.grip_target is not None else 0.0
        poses[slot] = {"q_deg": [round(float(v), 3) for v in np.degrees(q)] + [round(grip, 3)],
                       "time": time.strftime("%H:%M:%S")}
        path.write_text(json.dumps(poses, indent=1))
        self.msg = f"saved pose {slot} -> {path.name}"
        print(f"\n[pose {slot}] {fmt_vec(np.degrees(q))}", flush=True)


def main():
    args = parse()
    setup_logging(args.verbose)
    if abs(args.hz / args.fps - round(args.hz / args.fps)) > 1e-6:
        raise SystemExit(f"--hz {args.hz} must be a multiple of --fps {args.fps}")
    arm = make_follower(args)
    arm.cfg.gravity_compensation = True   # builds the URDF gravity model at connect; gravity gains for hold
    arm.cfg.max_step_deg = None           # free mode moves far between commands: no step clamp on quit/hold
    rig, saver = None, None
    if args.out:
        from .record import Saver, write_meta
        out = Path(args.out)
        for split in ("train", "val"):
            (out / split).mkdir(parents=True, exist_ok=True)
        rig = make_rig(args).start()
        saver = Saver(out, args.val_every)
        args.instruction = " | ".join(args.instr_list)
        write_meta(out, args, arm, rig)
    loop = RateLoop(args.hz)
    fr = Free(arm, args, rig, saver)
    keys = KeyReader()
    try:
        arm.connect(calibrate=not args.skip_gripper_check)
        if arm._gravity_pairing is None:
            raise RuntimeError("gravity model not available: refusing free mode")
        print(__doc__.split("Starts in")[1].split("Safety")[0], flush=True)
        print(f"logging to {fr.open_log()}", flush=True)
        fr.msg = "LOCKED. hold the arm + press f to release" + (", h = ready pose" if args.ready else "")
        loop.reset()
        tick, t0 = 0, time.perf_counter()
        while True:
            now = time.perf_counter() - t0
            q, qd, ff = fr.tick(now, tick)
            for k in keys.get():
                if k == "q":
                    raise KeyboardInterrupt
                fr.key(k, now, q)
            if tick % int(args.hz * 0.3) == 0 and fr.mode != "REPLAY":  # replay: arm.command checks temps
                arm._check_motor_temperatures(arm.cfg.temp_alarm_threshold_c, arm.cfg.temp_overheat_threshold_c,
                                              "free")
            if tick % int(args.hz / 10) == 0:
                tmax = max(arm.last_temps.values()) if arm.last_temps else float("nan")
                rec = f" REC {now - fr.rec['t0']:4.1f}s" if fr.rec else ""
                ins = f" | '{fr.instruction[:24]}'" if saver else ""
                qs = " ".join(f"{v:4.0f}" for v in np.degrees(q))
                line = (f"[{fr.mode}{rec}] {fr.msg[:62]}{ins} | q {qs} | {loop.overruns} over | {tmax:.0f}C")
                width = shutil.get_terminal_size((160, 24)).columns - 1
                print("\r\033[K" + line[:width], end="", flush=True)
            tick += 1
            loop.wait()
    except KeyboardInterrupt as e:
        print(f"\nshutting down ({e or 'quit'})...", flush=True)
    finally:
        keys.close()
        if arm.is_connected:
            try:  # stiff hold where it is for a moment, then the plugin's slow return to zero
                arm.cfg.max_step_deg = None
                arm.cfg.gravity_stiff = False
                for _ in range(15):
                    arm.hold()
                    time.sleep(0.02)
            except Exception as e:
                print(f"hold before shutdown failed: {e}")
            arm.disconnect(safe_zero=True)
        if fr.log:
            fr.log.close()
        if rig:
            rig.stop()
        if saver:
            saver.close()


if __name__ == "__main__":
    main()
