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
            "--samples_file", str(tmp_path / "samples.json"), "--log_dir", str(tmp_path / "logs")]
    out = _run("b601free.free", args, [(2, "f"), (4, "r"), (3, "s"), (12, "1"), (1, "q")])
    assert "Traceback" not in out, out[-2000:]
    eps = sorted((tmp_path / "data").glob("*/ep_*.npz"))
    assert len(eps) == 1, out[-2000:]
    z = np.load(eps[0])
    T = len(z["q"])
    assert T > 60 and z["front"].shape == (T, 256, 256, 3) and z["action"].shape == (T, 7)
    assert str(z["instruction"]) == "hit the red bar"
    assert "1" in json.loads((tmp_path / "poses.json").read_text())
    assert list((tmp_path / "logs").glob("free_*.csv"))


def test_xylophone_player(tmp_path):
    poses = {str(i): {"q_deg": [-15 + 10 * (i - 1), 40, 60, -30, 0, 0, 0]} for i in range(1, 4)}
    poses["0"] = {"q_deg": [-15, 40, 60, -38, 0, 0, 0]}
    poses["9"] = {"q_deg": [0, 55, 60, -10, 0, 0, 0]}
    (tmp_path / "poses.json").write_text(json.dumps(poses))
    out = _run("b601free.xylo", ["--poses", str(tmp_path / "poses.json"), "--gravity_scales", "/nonexistent"],
               [(5, "1"), (0.2, "3"), (2, "q")])
    assert "Traceback" not in out, out[-2000:]
    assert "hit 1" in out and "hit 3" in out


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d)) if fn.__code__.co_argcount else fn()
            print("ok", name)
