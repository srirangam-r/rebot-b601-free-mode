"""Teleoperate the B601-RS follower from the reBot Arm 102 leader at 30 Hz and record demo episodes in the
sim format (sim/gen_data.py): <out>/{train,val}/ep_XXXX_<time>.npz with
  front, wrist  (T,256,256,3) uint8 RGB
  q, qvel, tau  (T,7) float32   follower motor-frame deg, deg/s, N*m (6 arm joints + gripper)
  action        (T,7) float32   motor-frame deg targets actually sent to the follower at that tick
  instruction   str
Frame t pairs the observation read at the start of tick t with the command sent in tick t.

Keys (terminal):  r/enter = start episode   s = stop+save   d = stop+discard   q/esc = quit
Ctrl-C = quit now (an unsaved episode is discarded). On quit the follower runs the plugin's safe_zero
(slow return to the zero pose) and then disables torque; Ctrl-C during that disables immediately.

  python -m b601free.record --out data/real_cube --instruction "put the red cube on the green pad" \
      --front /dev/v4l/by-id/...front...-index0 --wrist /dev/v4l/by-id/...wrist...-index0
  python -m b601free.record --mock --out /tmp/x --instruction test    # no hardware
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .b601 import GRIPPER, Overheat, add_follower_args, make_follower
from .cameras import add_camera_args, make_rig
from .common import JOINTS, RateLoop, setup_logging
from .keys import KeyReader
from .leader import add_leader_args, make_leader
from .preview import Preview

logger = logging.getLogger("record")
EP_RE = re.compile(r"ep_(\d{4})_")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="dataset dir (episodes go to <out>/train and <out>/val)")
    p.add_argument("--instruction", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_ep_s", type=float, default=90.0, help="recording stops here; then press s or d")
    p.add_argument("--min_frames", type=int, default=15)
    p.add_argument("--val_every", type=int, default=10, help="episode index %% N == N-1 goes to val/ (0 = never)")
    p.add_argument("--max_step", type=float, default=6.0, help="arm: max target change per tick [deg]")
    p.add_argument("--max_step_grip", type=float, default=60.0, help="gripper: max change per tick [motor deg]")
    p.add_argument("--align_step", type=float, default=1.0, help="start-up ramp to the leader pose [deg/tick]")
    p.add_argument("--no_cameras", action="store_true", help="teleop only, recording disabled")
    p.add_argument("--no_preview", action="store_true")
    p.add_argument("--mock", action="store_true", help="fake arm, leader and cameras")
    p.add_argument("-v", "--verbose", action="store_true")
    add_follower_args(p)
    add_leader_args(p)
    add_camera_args(p)
    p.add_argument("--skip_gripper_check", action="store_true", help="skip the connect-time gripper close-to-zero check (use when the gripper holds a tool, e.g. a mallet; the gripper zero itself lives in the motor)")
    return p.parse_args()


class Saver:
    """Background episode writer (np.savez_compressed takes seconds; the control loop must not stall).
    Writes to a hidden temp file then renames, so a crash never leaves a truncated ep_*.npz."""

    def __init__(self, out: Path, val_every: int):
        self.out, self.val_every = out, val_every
        existing = [int(m.group(1)) for f in out.glob("*/ep_*.npz") if (m := EP_RE.match(f.name))]
        self.next_idx = max(existing, default=-1) + 1
        self.q: queue.Queue = queue.Queue()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        self.n_saved = 0
        self.n_submitted = 0
        self.n_done = 0

    def submit(self, ep: dict, info: dict) -> Path:
        idx = self.next_idx
        self.next_idx += 1
        split = "val" if self.val_every > 0 and idx % self.val_every == self.val_every - 1 else "train"
        path = self.out / split / f"ep_{idx:04d}_{datetime.now():%Y%m%d-%H%M%S}.npz"
        self.q.put((path, ep, info))
        self.n_submitted += 1
        return path

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, ep, info = item
            try:
                t0 = time.time()
                instruction = ep.pop("instruction")
                ep = {k: np.stack(v).astype(np.uint8 if k in ("front", "wrist") else np.float32) for k, v in ep.items()}
                ep["instruction"] = np.array(instruction)
                tmp = path.with_name("." + path.name + ".partial")
                with open(tmp, "wb") as f:
                    np.savez_compressed(f, **ep)
                tmp.replace(path)
                info |= {"file": str(path.relative_to(self.out)), "save_s": round(time.time() - t0, 2)}
                with open(self.out / "episodes.jsonl", "a") as f:
                    f.write(json.dumps(info) + "\n")
                self.n_saved += 1
                print(f"\n[saved] {path} ({info['frames']} frames, {info['save_s']} s to write)", flush=True)
            except Exception as e:
                logger.exception("saving %s failed: %s", path, e)
            finally:
                self.n_done += 1

    def close(self):
        if self.n_done < self.n_submitted:
            print("\nwaiting for episode writes to finish...", flush=True)
        self.q.put(None)
        self.th.join()


def write_meta(out: Path, args, arm, rig):
    meta_path = out / "meta.json"
    meta = {
        "format": "sim/gen_data.py npz: front,wrist (T,256,256,3) uint8 RGB; q,qvel,tau,action (T,7) float32; "
                  "instruction str",
        "joints": JOINTS,
        "units": {"q": "deg (follower motor frame = plugin get_observation .pos)", "qvel": "deg/s",
                  "tau": "N*m (motor-reported)", "action": "deg, motor frame, target sent to the follower"},
        "fps": args.fps,
        "gripper": {"closed_motor_deg": 0.0, "fully_open_motor_deg": arm.cfg.joint_limits[GRIPPER][1],
                    "note": "follower gripper motor deg = 6 x leader gripper deg, clipped to [0, 270]; 0 = closed "
                            "(zeroed closed, checked by detect_gripper_zero), larger = more open"},
        "follower": {"joint_limits": arm.cfg.joint_limits, "joint_directions": arm.cfg.joint_directions,
                     "gravity_compensation": arm.cfg.gravity_compensation, "max_step_deg": arm.cfg.max_step_deg},
        "cameras": rig.info() if rig else None,
        "instruction": args.instruction,
        "mock": args.mock,
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("instruction") != args.instruction:
            logger.warning("this dataset was recorded with instruction %r; now %r (fine for multi-task data; for a "
                           "single-task trainer use a separate --out per task).", old.get("instruction"), args.instruction)
        if old.get("cameras") != meta["cameras"]:
            logger.warning("camera config differs from earlier sessions in this dir: %s vs %s",
                           old.get("cameras"), meta["cameras"])
    meta_path.write_text(json.dumps(meta, indent=1))


def align(arm, leader, loop, step_deg: float, timeout_s: float = 20.0):
    """Ramp the follower to the leader pose slowly before normal teleop (the plugin jumps straight there)."""
    saved = arm.cfg.max_step_deg
    arm.cfg.max_step_deg = {j: step_deg * (6.0 if j == GRIPPER else 1.0) for j in JOINTS}
    print(f"aligning follower to leader at {step_deg * loop.period ** -1:.0f} deg/s ...", flush=True)
    t0 = time.time()
    try:
        while True:
            arm.read()
            m = arm.action_to_motor(leader.get_action())
            tgt = np.array([m[j] for j in JOINTS])
            sent = arm.command(tgt)
            err = np.abs(sent - tgt)
            if err.max() < 1e-6:
                break
            if time.time() - t0 > timeout_s:
                logger.warning("align timeout (leader moving?) - continuing with the normal step limit")
                break
            print(f"\r  remaining {err.max():6.1f} deg ({JOINTS[int(err.argmax())]})   ", end="", flush=True)
            loop.wait()
    finally:
        arm.cfg.max_step_deg = saved
    print("\raligned.                                   ", flush=True)


def main():
    args = parse()
    setup_logging(args.verbose)
    out = Path(args.out)
    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)

    leader = make_leader(args)
    arm = make_follower(args)
    arm.cfg.max_step_deg = {j: args.max_step for j in JOINTS[:6]} | {GRIPPER: args.max_step_grip}
    rig = None if args.no_cameras else make_rig(args)
    preview = Preview(enabled=not args.no_preview and rig is not None, title=f"record: {args.instruction}")
    saver = Saver(out, args.val_every)
    keys = KeyReader()
    loop = RateLoop(args.fps)
    state, ep, ep_t0, ep_frames = "IDLE", None, 0.0, 0
    stats = {"max_dt": 0.0, "over": 0}
    line = ""
    try:
        leader.connect(calibrate=True)
        if rig:
            rig.start()
        write_meta(out, args, arm, rig)
        preview.start()
        arm.connect(calibrate=not args.skip_gripper_check)
        loop.reset()
        align(arm, leader, loop, args.align_step)
        print("\nkeys: r/enter = start episode | s = save | d = discard | q = quit"
              + ("" if rig else "   (no cameras: teleop only)"), flush=True)
        print(f"next episode index: {saver.next_idx}  ->  {out}", flush=True)
        loop.reset()
        tick = 0
        while True:
            t_tick = time.perf_counter()
            obs = arm.read()
            if rig:
                front, wrist, ages = rig.read()
            m = arm.action_to_motor(leader.get_action())
            target = np.array([m[j] for j in JOINTS])
            sent = arm.command(target)

            if state == "REC":
                for k, v in (("front", front), ("wrist", wrist), ("q", obs["q"]), ("qvel", obs["qvel"]),
                             ("tau", obs["tau"]), ("action", sent)):
                    ep[k].append(v)
                ep["t"].append(t_tick)
                ep_frames += 1
                if time.perf_counter() - ep_t0 >= args.max_ep_s:
                    state = "FULL"
                    print(f"\nmax episode length {args.max_ep_s:.0f}s reached: s = save, d = discard", flush=True)

            for key in keys.get():
                if key in ("r", "enter") and state == "IDLE":
                    if rig is None:
                        print("\nrecording disabled (--no_cameras)", flush=True)
                        continue
                    ep = {k: [] for k in ("front", "wrist", "q", "qvel", "tau", "action", "t")}
                    ep_t0, ep_frames, state = time.perf_counter(), 0, "REC"
                    stats = {"max_dt": 0.0, "over": loop.overruns, "leader_fail": 0}
                    print(f"\n[REC] episode {saver.next_idx} started", flush=True)
                elif key == "s" and state in ("REC", "FULL"):
                    if ep_frames < args.min_frames:
                        print(f"\nonly {ep_frames} frames (< {args.min_frames}): discarded", flush=True)
                    else:
                        ts = np.array(ep.pop("t"))
                        arr = dict(ep, instruction=args.instruction)  # stacked in the saver thread
                        dts = np.diff(ts) if len(ts) > 1 else np.array([loop.period])
                        g = np.array([a[6] for a in ep["action"]])
                        info = {"frames": ep_frames, "seconds": round(float(ts[-1] - ts[0]), 2),
                                "mean_hz": round(float(1 / dts.mean()), 2), "max_dt_ms": round(float(dts.max() * 1e3), 1),
                                "overruns": loop.overruns - stats["over"], "leader_failures": stats.get("leader_fail", 0),
                                "grip_action_min": round(float(g.min()), 1), "grip_action_max": round(float(g.max()), 1),
                                "instruction": args.instruction, "time": datetime.now().isoformat(timespec="seconds")}
                        path = saver.submit(arr, info)
                        print(f"\n[STOP] {ep_frames} frames -> saving {path.name} ({path.parent.name}); "
                              f"gripper target {info['grip_action_min']}..{info['grip_action_max']} motor deg",
                              flush=True)
                    ep, state = None, "IDLE"
                elif key == "d" and state in ("REC", "FULL"):
                    print(f"\n[DISCARD] {ep_frames} frames dropped", flush=True)
                    ep, state = None, "IDLE"
                elif key in ("q", "esc"):
                    if state == "IDLE":
                        raise SystemExit(0)
                    print("\nepisode in progress: press s (save) or d (discard) first", flush=True)
            if leader.consecutive_failures and state == "REC":
                stats["leader_fail"] = stats.get("leader_fail", 0) + 1

            tick += 1
            if tick % 6 == 0:
                dt_ms = loop.last_dt * 1e3
                age = f"cam {ages[0] * 1e3:3.0f}/{ages[1] * 1e3:3.0f}ms" if rig else "no cam"
                lead = f"LEADER LOST x{leader.consecutive_failures}" if leader.consecutive_failures else "leader ok"
                rec = f"[{state} {ep_frames / args.fps:5.1f}s {ep_frames:4d}fr]" if state != "IDLE" else "[IDLE]"
                tmax = max(arm.last_temps.values()) if arm.last_temps else float("nan")
                line = (f"{rec} ep {saver.next_idx} | dt {dt_ms:4.1f}ms over {loop.overruns} | {age} | {lead} | "
                        f"T {tmax:4.0f}C | grip tgt {sent[6]:5.1f} q {obs['q'][6]:5.1f}")
                print("\r" + line + "   ", end="", flush=True)
            if rig:
                preview.show(front, wrist, line)
            loop.wait()
    except SystemExit:
        pass
    except Overheat:
        print("\n[OVERHEAT] a motor exceeded the overheat threshold; shutting down.", flush=True)
    except KeyboardInterrupt:
        if ep_frames and state in ("REC", "FULL"):
            print(f"\nCtrl-C: unsaved episode ({ep_frames} frames) discarded", flush=True)
        print("\nCtrl-C", flush=True)
    finally:
        keys.close()
        preview.close()
        print("\nshutting down...", flush=True)
        try:
            arm.disconnect()
        finally:
            leader.disconnect()
            if rig:
                rig.stop()
            saver.close()
    print(f"done: {saver.n_saved} episode(s) saved this session in {out}")


if __name__ == "__main__":
    sys.exit(main())
