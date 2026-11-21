"""Play the xylophone from the keyboard. Scripted (no model): low latency, exact timing, repeatable.

    python -m b601free.xylo --skip_gripper_check

Poses come from `python -m b601free.free` (digit keys save slots in b601free/poses.json):
  slots 1-8  mallet ~1-2 cm above the centre of each bar (low -> high)
  slot 0     mallet resting on bar 1 (sets the strike direction + depth: strike = hover + depth x (slot0 - slot1))
  slot 9     safe pose well above the xylophone (start/quit go through it, so nothing sweeps across the bars)
Keys:
  1-8 or a s d f g h j k   strike that bar (notes queue up while the arm is busy)
  [ / ]    strike depth -/+ 0.1 (1.0 = exactly to the contact pose; >1 = drive through it)
  - / =    strike time -/+ 10 ms (shorter = harder hit)
  p        play a test melody ("Mary had a little lamb" on bars 3 2 1 2 3 3 3)
  q        quit: back to the safe pose, slow return to zero, torque off
Controller: stiff default gains + the calibrated gravity model (b601free.free calibration), commands at --hz.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .b601 import GRIPPER, Overheat, add_follower_args, make_follower
from .common import ARM_JOINTS, JOINTS, PKG, RateLoop, setup_logging
from .keys import KeyReader

PIANO = {k: str(i + 1) for i, k in enumerate("asdfghjk")}
MARY = "3212333 222 355 3212333 32213"  # spaces = rests (one beat)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_follower_args(p)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--skip_gripper_check", action="store_true", help="gripper holds the mallet")
    p.add_argument("--poses", default=str(PKG / "poses.json"))
    p.add_argument("--hz", type=float, default=100.0)
    p.add_argument("--move_speed", type=float, default=120.0, help="bar-to-bar travel speed [deg/s, worst joint]")
    p.add_argument("--min_move_s", type=float, default=0.12, help="shortest bar-to-bar move")
    p.add_argument("--goto_speed", type=float, default=30.0, help="slow moves (start, quit) [deg/s]")
    p.add_argument("--depth", type=float, default=1.2, help="strike depth (x the hover->contact offset)")
    p.add_argument("--down_s", type=float, default=0.08, help="strike down-stroke time")
    p.add_argument("--up_s", type=float, default=0.10, help="strike up-stroke time")
    p.add_argument("--beat_s", type=float, default=0.45, help="test-melody beat")
    p.add_argument("--max_err", type=float, default=15.0, help="stop if a joint is this far off for 10 ticks [deg]")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def minjerk(u):
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


class Player:
    def __init__(self, arm, args, poses):
        self.arm, self.a = arm, args
        self.hover = {s: np.array(poses[s]["q_deg"][:6]) for s in map(str, range(1, 9)) if s in poses}
        if not self.hover:
            raise SystemExit(f"no bar poses (slots 1-8) in {args.poses}: teach them with `python -m b601free.free`")
        if "0" not in poses or "1" not in poses:
            raise SystemExit("need slot 0 (mallet touching bar 1) and slot 1 (hover over bar 1) for the strike")
        self.strike_vec = np.array(poses["0"]["q_deg"][:6]) - self.hover["1"]
        self.safe = np.array(poses["9"]["q_deg"][:6]) if "9" in poses else None
        self.segs = deque()          # (q_from, q_to, T, profile, tag)
        self.cur = None              # active segment + start time
        self.target = None           # deg, 6
        self.notes = deque()
        self.at = None               # bar slot the arm is hovering over
        self.bad = 0
        self.hits = []
        self.msg = ""
        print(f"bars: {sorted(self.hover)} | strike offset (deg) {np.round(self.strike_vec, 1)} "
              f"| safe pose: {'slot 9' if self.safe is not None else 'NONE (teach slot 9!)'}")

    def move_time(self, frm, q_to, speed=None, min_s=None):
        d = float(np.abs(q_to - frm).max())
        if d < 0.5 and min_s is None:   # already there: no move (lowest latency for repeated notes)
            return 0.0
        return max(min_s if min_s is not None else self.a.min_move_s, d / (speed or self.a.move_speed))

    def plan_move(self, q_to, speed=None, min_s=None):
        frm = self.segs[-1][1] if self.segs else self.target
        T = self.move_time(frm, q_to, speed, min_s)
        if T > 0:
            self.segs.append((frm.copy(), q_to.copy(), T, "minjerk", "move"))

    def plan_note(self, slot, now=None, hit_at=None):
        h = self.hover[slot]
        if hit_at is not None:   # melody: wait so the mallet lands on the beat
            lead = self.move_time(self.target, h) + self.a.down_s
            if hit_at - lead > now:
                self.segs.append((self.target.copy(), self.target.copy(), hit_at - lead - now, "minjerk", "wait"))
        self.plan_move(h)
        hit = h + self.a.depth * self.strike_vec
        self.segs.append((h.copy(), hit, self.a.down_s, "in", f"hit {slot}"))   # accelerate into the bar
        self.segs.append((hit, h.copy(), self.a.up_s, "out", "up"))            # snap back
        self.at = slot

    def step(self, now):
        """Advance the trajectory; returns the 6 arm targets [deg]."""
        if self.cur is None and not self.segs and self.notes:
            n, hit_at = self.notes.popleft()
            if n != " ":
                self.plan_note(n, now, hit_at)
        if self.cur is None and self.segs:
            self.cur = (self.segs.popleft(), now)
        if self.cur is not None:
            (frm, to, T, prof, tag), t0 = self.cur
            u = min(1.0, (now - t0) / T)
            s = minjerk(u) if prof == "minjerk" else (u * u if prof == "in" else 1 - (1 - u) ** 2)
            self.target = frm + s * (to - frm)
            if u >= 1.0:
                self.cur = None
                if tag.startswith("hit"):
                    self.msg = f"{tag} at {now:.2f}s"
                    self.hits.append(now)
        return self.target

    @property
    def busy(self):
        return self.cur is not None or bool(self.segs) or bool(self.notes)


def main():
    raise SystemExit("b601free.xylo is retired: its automatic start move from home is not limited to the taught area.\n"
                     "Use free mode instead:  python -m b601free.free   -> hand-guide the arm near the bars (f) -> m\n"
                     "(music mode only moves inside the box of your taught poses and stops on any collision).")
    args = parse()
    setup_logging(args.verbose)
    poses = json.loads(Path(args.poses).read_text()) if Path(args.poses).is_file() else {}
    arm = make_follower(args)
    arm.cfg.gravity_compensation = True   # calibrated gravity model (b601free/gravity_fit.json) as feed-forward
    arm.cfg.gravity_stiff = True          # ... on top of the stiff default gains: accurate fast tracking
    arm.cfg.max_step_deg = {j: 12.0 for j in ARM_JOINTS} | {GRIPPER: 30.0}   # per-tick safety cap (1200 deg/s)
    pl = Player(arm, args, poses)
    keys, loop = KeyReader(), RateLoop(args.hz)
    try:
        arm.connect(calibrate=not args.skip_gripper_check)
        obs = arm.read()
        grip = float(obs["q"][6])
        pl.target = obs["q"][:6].copy()
        if pl.safe is not None:
            pl.plan_move(pl.safe, speed=args.goto_speed, min_s=1.0)
        pl.plan_move(pl.hover[min(pl.hover)], speed=args.goto_speed, min_s=1.0)
        pl.at = min(pl.hover)
        print(__doc__.split("Keys:")[1].split("Controller")[0], flush=True)
        pl.msg = "moving to bar 1 ..."
        t0 = time.perf_counter()
        loop.reset()
        tick = 0
        while True:
            now = time.perf_counter() - t0
            obs = arm.read()
            tgt = pl.step(now)
            err = np.abs(obs["q"][:6] - pl.target)
            pl.bad = pl.bad + 1 if err.max() > args.max_err else 0
            if pl.bad >= 10:
                raise RuntimeError(f"{ARM_JOINTS[int(err.argmax())]} {err.max():.0f} deg off target (blocked?)")
            arm.command(np.append(tgt, grip))
            for k in keys.get():
                k = PIANO.get(k, k)
                if k == "q":
                    raise KeyboardInterrupt
                if k in pl.hover:
                    pl.notes.append((k, None))       # live key: as soon as possible
                elif k.isdigit():
                    pl.msg = f"bar {k} not taught"
                elif k in ("[", "]"):
                    args.depth = round(max(0.5, args.depth + (0.1 if k == "]" else -0.1)), 2)
                    pl.msg = f"depth {args.depth:.1f}"
                elif k in ("-", "="):
                    args.down_s = round(min(0.3, max(0.03, args.down_s + (0.01 if k == "=" else -0.01))), 3)
                    pl.msg = f"strike time {args.down_s * 1e3:.0f} ms"
                elif k == "p":
                    t_first = now + 1.0
                    pl.notes.extend((c, t_first + i * args.beat_s) for i, c in enumerate(MARY)
                                    if c == " " or c in pl.hover)
                    pl.msg = f"playing test melody at {60 / args.beat_s:.0f} bpm"
            if tick % int(args.hz / 10) == 0:
                tmax = max(arm.last_temps.values()) if arm.last_temps else float("nan")
                print(f"\r\033[K[{'BUSY' if pl.busy else 'READY'}] bar {pl.at} | queue {len(pl.notes)} | "
                      f"depth {args.depth:.1f} | strike {args.down_s * 1e3:.0f} ms | {loop.overruns} over | "
                      f"{tmax:.0f}C | {pl.msg}", end="", flush=True)
            tick += 1
            loop.wait()
    except (KeyboardInterrupt, Overheat, RuntimeError) as e:
        print(f"\nstopping ({e or 'quit'})...", flush=True)
    finally:
        keys.close()
        if arm.is_connected:
            try:  # leave through the safe pose so the mallet never sweeps across the bars
                if pl.safe is not None and pl.target is not None:
                    pl.segs.clear(), pl.notes.clear()
                    pl.cur = None
                    pl.plan_move(pl.safe, speed=args.goto_speed, min_s=1.0)
                    t1 = time.perf_counter()
                    while pl.busy:
                        arm.read()
                        arm.command(np.append(pl.step(time.perf_counter() - t1), grip))
                        time.sleep(1.0 / args.hz)
                arm.cfg.gravity_stiff = False
                arm.cfg.max_step_deg = None
            except Exception as e:
                print(f"move to the safe pose failed: {e}")
            arm.disconnect(safe_zero=True)


if __name__ == "__main__":
    main()
