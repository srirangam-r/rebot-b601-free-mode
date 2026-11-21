# rebot-b601-free-mode

Hand-guided **free mode** for the Seeed **reBot B601-RS** arm (RobStride motors over CAN), without LeRobot. It has
gravity and friction compensation calibrated on your own arm, kinesthetic demo recording with automatic **replay**
(so the training images don't contain your hand), leader-arm teleoperation, and a keyboard xylophone player.

Built in one day at a hackathon. The driver, teleop, free mode and calibration have run on a real B601-RS.
Demo replay and the xylophone player have only run in mock mode.

## What's in it

| module | what |
|---|---|
| `b601free/b601.py` | follower driver (MIT mode, gains, joint limits, temperature protection, safe two-stage return to zero), ported from Seeed's LeRobot plugin. Adds per-command step caps, a stiff-plus-gravity mode, fitted gravity models and a gripper lock. `python -m b601free.b601 read` |
| `b601free/free.py` | **free mode**: zero stiffness, gravity and friction feed-forward, automatic calibration, pose teaching, demo recording and replay |
| `b601free/gravity.py` | URDF gravity model (Seeed's, vectorised) plus a **regressor** `g(q) = Y(q)·θ` for fitting link masses and centres of mass |
| `b601free/record.py` | 30 Hz leader-follower teleop and episode recording (reBot Arm 102 leader) |
| `b601free/xylo.py` | keyboard xylophone player built on taught poses (scripted, with on-the-beat melody timing) |
| `b601free/check.py`, `leader.py`, `cameras.py`, `preview.py`, `keys.py`, `mock.py`, `common.py` | joint-direction check, leader reader, threaded UVC cameras, Tk preview, terminal keys, hardware fakes, helpers |

## Install

```bash
git clone <this repo> && cd rebot-b601-free-mode
python -m venv .venv && source .venv/bin/activate
pip install -e .            # numpy, opencv-python, motorbridge, motorbridge-smart-servo
```

CAN (PCAN-USB or another SocketCAN adapter), once per boot or replug:
```bash
sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 txqueuelen 1000 && sudo ip link set can0 up
ip -details link show can0    # want ERROR-ACTIVE, bitrate 1000000
```
The leader is a USB-serial servo bus on `/dev/ttyUSB0`; your user needs to be in the `dialout` group.

Everything also runs **without hardware** with `--mock`, and `python tests/test_mock.py` checks the whole stack.

## Units
`q`, actions and joint limits are in **follower motor-frame degrees**, exactly what Seeed's plugin reports
(`math.degrees(state.pos)`). `qvel` is in deg/s and `tau` in N·m (motor-reported). The URDF takes the same raw motor
angles in radians: joint1..joint6 are pan, lift, elbow, wrist_flex, wrist_yaw, wrist_roll.

## Gripper lock (read this)
The gripper is **locked by default** (`B601Config.gripper_locked = True`). It gets zero torque every tick, skips the
connect-time close-to-zero check, and is left out of the return to zero; recorded gripper actions are its measured
position. This setup had a mallet taped into the gripper. For normal grasping, pass `--unlock_gripper`.

## Free mode

```bash
python -m b601free.free
```
It starts **locked** (HOLD: a soft position hold plus gravity). Hold the arm and press `f`: over 3 s the stiffness
ramps to zero while the gravity torque takes over, and the arm floats.

| key | action |
|---|---|
| `f` | release (FREE) / lock where it is (HOLD) |
| `a` | **automatic calibration** from the current pose (see below) |
| `0`-`9` | save the current pose to slot N (`b601free/poses.json`) |
| `h` / `z` | move by itself to the `--ready` pose / home (pan and wrist first, then lift and elbow) |
| `r` / `s` / `d` | record a demo (from FREE) / stop, replay and save / discard |
| `i` | next instruction (`--instructions "a\|b\|c"`) |
| `j` `+` `-` | select a joint / nudge its gravity scale |
| `q` | quit: hold, slow return to zero, torque off |

Safety:
- Soft springs near the joint limits, except at the lift and elbow rest stops.
- A speed watchdog (150 deg/s for the big joints, 400 for the wrist) and a drift check during the release ramp.
- Moves stop if a joint is blocked.
- Motor temperatures are checked every 0.3 s.
- A 30 Hz CSV log of each session goes to `b601free/logs/`.

### How the calibration works
1. Press `a` from a pose up in free space. The arm moves itself (15 deg/s) through 9 poses: lift ±20, elbow ±25,
   wrist ±30, plus two combinations.
2. At each pose it holds, waits until every joint has moved less than 0.4° in 0.4 s, and averages the motor-reported
   torque for 0.5 s. That is the torque gravity needs there.
3. Gravity torque is **linear in each link's mass and mass × centre-of-mass**, so it fits those parameters by ridge
   least squares on **every sample collected so far** (`gravity_samples.json`, kept across runs), pulled toward the
   URDF values. Ordinary links may move about 25%; the tool link gets about ±0.4 kg of freedom for a camera or tool
   payload. The result goes in `gravity_fit.json`, which free mode, replay and the driver's `--gravity_comp` all load.
4. While the arm moves between poses, measured torque minus gravity torque gives each joint's **Coulomb friction**.
   Free mode cancels 70% of it in the direction of motion, above 2 deg/s.

Run `a` in each region you'll use, such as near home and near the work area. On the arm this was built on,
18 samples halved the gravity error versus the raw URDF (elbow 0.39 → 0.18 N·m mean).

### Recording demos (kinesthetic, with replay)
```bash
python -m b601free.free --out data/task --ready 1 --instructions "hit the red bar|hit the blue bar" \
    --front /dev/v4l/by-id/<overhead>-video-index0 --wrist /dev/v4l/by-id/<wrist>-video-index0
```
1. Press `h` to go to the ready pose, `f` to release, then `r`, guide the motion, and `s`.
2. The arm moves back to where the demo started, waits `--wait_s` (step back), and **replays your motion** with the
   stiff controller (`--gravity_comp --gravity_stiff`).
3. During the replay it records both cameras at `--fps`. The action is your trajectory `--lead_ms` ahead; q, qvel and
   tau are what the arm actually did. No hand appears in the images, and the actions are exactly what this
   controller can reproduce.
4. A `[replay]` line reports tracking error versus your demo. `--no_replay` saves the hand-guided pass itself instead.

Episodes are saved as `<out>/{train,val}/ep_XXXX_<time>.npz` with `front` and `wrist` (T, 256, 256, 3, uint8 RGB),
`q`, `qvel`, `tau` and `action` (T, 7, float32) and `instruction`, plus `episodes.jsonl` and `meta.json`.

## Leader teleop recording
```bash
python -m b601free.check                     # torque off: leader vs follower angles, to check joint directions
python -m b601free.record --out data/task --instruction "..." --front <src> --wrist <src>
```
Keys: `r` start, `s` save, `d` discard, `q` quit.

Note: with torque **off**, the follower's position only updates at connect, because motorbridge's
`request_feedback()` sends nothing for RobStride motors. So `check` / `read` show the follower's position at connect
time. With torque on, every command returns fresh feedback.

## Xylophone player
Teach poses in free mode: slots 1-8 hover about 1-2 cm above each bar, 0 with the mallet resting on bar 1 (sets the
strike direction and depth), 9 for a safe pose above the instrument. Then:
```bash
python -m b601free.xylo
```
| key | action |
|---|---|
| `1`-`8` or `a s d f g h j k` | strike that bar |
| `[` `]` | strike depth |
| `-` `=` | down-stroke time |
| `p` | test melody, played on the beat |
| `q` | quit through the safe pose |

A rigidly held mallet sounds dead. Mount it on a pivot or flexible shaft so it bounces off the bar.

## Cameras
`python -m b601free.cameras --list`; `--snap <src> <src>` writes `/tmp/cam_*.png`. Prefer
`/dev/v4l/by-id/...-index0` paths. `scripts/cam_setup.sh` locks exposure and white balance and sets 60 Hz
anti-flicker; edit the device paths for your cameras.

## Credits and license
MIT (see `LICENSE`). The motor driver, gravity model and URDF are ported from Seeed Studio's MIT-licensed LeRobot
plugins (`lerobot_robot_seeed_b601` 1.0.5, `lerobot_teleoperator_rebot_arm_102` 1.0.1). See
`THIRD_PARTY_NOTICES.md`. Motor I/O goes through [motorbridge](https://github.com/motorbridge/motorbridge) (MIT).
