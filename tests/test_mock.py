"""Hardware-free checks (mock motors, servos and cameras). Run: python -m pytest tests  (or python tests/test_mock.py)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from b601free import b601 as B  # noqa: E402
from b601free.gravity import compute_generalized_gravity, gravity_regressor, link_params, load_dynamics_model  # noqa: E402


def _mock_arm():
    p = argparse.ArgumentParser()
    B.add_follower_args(p)
    p.add_argument("--mock", action="store_true")
    return B.make_follower(p.parse_args(["--mock", "--gravity_scales", "/nonexistent"]))


def test_regressor_matches_model():
    m = load_dynamics_model()
    th = link_params(m)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, m.nq)
        assert np.allclose(compute_generalized_gravity(m, q), gravity_regressor(m, q) @ th, atol=1e-9)


def test_gripper_is_never_driven():
    arm = _mock_arm()
    assert arm.cfg.gripper_locked
    arm.connect(calibrate=True)                  # the close-to-zero probe must not run while locked
    act = arm.command(np.array([5, 10, 10, 0, 0, 0, 200.0]))
    assert arm.bus.cmd[6] == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert abs(act[6]) < 5.0                      # recorded gripper action = measured position, not the 200 asked
    arm.hold()
    assert arm.bus.cmd[6] == (0.0, 0.0, 0.0, 0.0, 0.0)
    arm.disconnect(safe_zero=True)


def _run(module, args, keys, timeout=90):
    """Run `python -m module args` with scripted keys [(delay_s, key), ...] on stdin; return its output."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", COLUMNS="200", PYTHONPATH=str(REPO))
    p = subprocess.Popen([sys.executable, "-m", module, "--mock", *args], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=REPO)
    for delay, k in keys:
        time.sleep(delay)
        p.stdin.write(k.encode())
        p.stdin.flush()
    out, _ = p.communicate(timeout=timeout)
    return out.decode(errors="ignore").replace("\r", "\n")


def test_free_mode_record_and_replay(tmp_path):
    zero = {j: 0.0 for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")}
    (tmp_path / "scales.json").write_text(json.dumps(zero))   # the mock has no gravity to cancel
    args = ["--out", str(tmp_path / "data"), "--instructions", "hit the red bar|hit the blue bar",
            "--gravity_scales", str(tmp_path / "scales.json"), "--poses", str(tmp_path / "poses.json"),
            "--samples_file", str(tmp_path / "samples.json"), "--log_dir", str(tmp_path / "logs"), "--overhead", ""]
    out = _run("b601free.free", args, [(2, "f"), (4, "r"), (3, "s"), (12, "t"), (0.3, "1"), (1, "q")])
    assert "Traceback" not in out, out[-2000:]
    eps = sorted((tmp_path / "data").glob("*/ep_*.npz"))
    assert len(eps) == 1, out[-2000:]
    z = np.load(eps[0])
    T = len(z["q"])
    assert T > 60 and z["front"].shape == (T, 256, 256, 3) and z["action"].shape == (T, 7)
    assert str(z["instruction"]) == "hit the red bar"
    assert "1" in json.loads((tmp_path / "poses.json").read_text())
    assert list((tmp_path / "logs").glob("free_*.csv"))


def _music_files(tmp_path, contact_offset):
    zero = {j: 0.0 for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")}
    (tmp_path / "scales.json").write_text(json.dumps(zero))
    h1, h8 = [40, 62, 20, 20, 0, 0], [58, 63, 22, 18, 0, 0]
    add = lambda p_, d: [x + y for x, y in zip(p_, d)]
    poses = {"1": {"q_deg": h1 + [0]}, "c1": {"q_deg": add(h1, contact_offset) + [0]},
             "8": {"q_deg": h8 + [0]}, "c8": {"q_deg": add(h8, [0, 1, 3, -6, 0, 0]) + [0]},
             "9": {"q_deg": [49, 55, 20, 15, 0, 0, 0]}}
    (tmp_path / "poses.json").write_text(json.dumps(poses))
    (tmp_path / "ref.json").write_text(json.dumps({"bars": [[0, i * 20.0, 100 - i] for i in range(8)],
                                                    "fractions": [i / 7 for i in range(8)], "time": "test"}))
    return ["--gravity_scales", str(tmp_path / "scales.json"), "--poses", str(tmp_path / "poses.json"),
            "--samples_file", str(tmp_path / "s.json"), "--log_dir", str(tmp_path / "logs"), "--ready", "9",
            "--xylo_ref", str(tmp_path / "ref.json"), "--overhead", ""]


def test_music_mode_plays_inside_box(tmp_path):
    args = _music_files(tmp_path, [0, 1, 3, -6, 0, 0])
    out = _run("b601free.free", args, [(2, "m"), (0.5, "h"), (7, "m"), (1.5, "1"), (0.1, "5"), (0.1, "8"),
                                        (3, " "), (0.3, "m"), (0.3, "q"), (0.3, "q")])
    assert "Traceback" not in out and "STOPPED" not in out, out[-2000:]
    assert "REFUSED: move the mallet near" in out            # entry from home is refused
    assert "camera-blended bars [2, 3, 4, 5, 6, 7]" in out
    assert all(f"hit {k}" in out for k in "158"), out[-2000:]


def test_music_mode_refuses_posture_change_strike(tmp_path):
    args = _music_files(tmp_path, [0, 3, 11, -17, -3, 0])    # contact 17 deg away from hover: not a short drop
    out = _run("b601free.free", args, [(2, "h"), (7, "m"), (1, "q"), (0.3, "q")])
    assert "Traceback" not in out, out[-2000:]
    assert "contact is 17 deg from hover" in out, out[-2000:]


def test_virtual_friction_stops_creep(tmp_path):
    """+0.3 N*m of gravity-model error on the lift joint: pure float drifts, virtual friction holds.
    (The mock has no gravity, so the gravity model is zeroed and the error is injected explicitly.)"""
    import math
    from b601free import free as F
    zero = {j: 0.0 for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")}
    (tmp_path / "scales.json").write_text(json.dumps(zero))
    drift = {}
    for vf in (False, True):
        sys.argv = ["free", "--mock", "--overhead", "", "--gravity_scales", str(tmp_path / "scales.json"),
                    "--samples_file", str(tmp_path / "s.json")] + ([] if vf else ["--no_vfric"])
        a = F.parse()
        arm = F.make_follower(a); arm.cfg.gravity_compensation = True; arm.cfg.max_step_deg = None
        fr = F.Free(arm, a); arm.connect(calibrate=False); ctl = arm.bus
        ctl.pos[:6] = np.radians([0, 30, 40, -20, 0, 0])
        t0 = time.perf_counter()
        for i in range(40):
            fr.tick(time.perf_counter() - t0, i); time.sleep(1 / 120)
        fr.mode = "FREE"; q0 = math.degrees(ctl.pos[1])
        for i in range(40, 280):
            fr.tick(time.perf_counter() - t0, i)
            c = list(ctl.cmd[1]); c[4] += 0.3; ctl.cmd[1] = tuple(c)
            time.sleep(1 / 120)
        drift[vf] = abs(math.degrees(ctl.pos[1]) - q0)
        arm.disconnect(safe_zero=False)
    assert drift[False] > 10.0 and drift[True] < 2.0, drift


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d)) if fn.__code__.co_argcount else fn()
            print("ok", name)
