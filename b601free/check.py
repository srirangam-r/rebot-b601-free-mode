"""Hardware sanity check, nothing moves: follower torque stays OFF, the leader is only read.

Prints, per joint: leader angle, the follower target it maps to (x direction, clipped), the follower's
measured angle, and their difference. Put the leader in some pose, move the (limp) follower BY HAND into
the same physical pose: every |diff| should be a few degrees. A diff that grows as you move a joint
means a wrong joint_directions sign/scale; a constant offset means a zero/calibration mismatch.

  python -m b601free.check            # leader + follower
  python -m b601free.check --no_leader / --no_follower
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .b601 import action_to_motor, add_follower_args, make_follower
from .common import JOINTS, setup_logging
from .leader import add_leader_args, make_leader


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_follower_args(p)
    add_leader_args(p)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--no_leader", action="store_true")
    p.add_argument("--no_follower", action="store_true")
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = until Ctrl-C)")
    args = p.parse_args()
    setup_logging()
    lead = None if args.no_leader else make_leader(args)
    arm = None if args.no_follower else make_follower(args)
    tty = sys.stdout.isatty()
    t0 = time.time()
    try:
        if lead:
            lead.connect(calibrate=True)
        if arm:
            arm.connect(calibrate=False, enable_torque=False)
        first = True
        while True:
            la = lead.read() if lead else np.full(7, np.nan)
            m = action_to_motor(la, arm.cfg if arm else make_follower(args).cfg) if lead else {}
            tgt = np.array([m.get(j, np.nan) for j in JOINTS])
            q = arm.read()["q"] if arm else np.full(7, np.nan)
            temps = arm.last_temps if arm else {}
            rows = [f"{'joint':<14}{'leader':>9}{'->target':>10}{'follower':>10}{'diff':>8}{'T[C]':>7}"]
            for i, j in enumerate(JOINTS):
                rows.append(f"{j:<14}{la[i]:9.1f}{tgt[i]:10.1f}{q[i]:10.1f}{q[i] - tgt[i]:8.1f}"
                            f"{temps.get(j, float('nan')):7.1f}")
            if arm and arm.missing:
                rows.append(f"NO FEEDBACK: {arm.missing}")
            if tty and not first:
                print(f"\033[{len(rows)}F", end="")
            print("\n".join(rows), flush=True)
            first = False
            if args.seconds and time.time() - t0 > args.seconds:
                break
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        if arm:
            arm.disconnect(safe_zero=False)  # torque was never enabled
        if lead:
            lead.disconnect()


if __name__ == "__main__":
    main()
