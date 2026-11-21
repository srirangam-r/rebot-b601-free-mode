# rebot-b601-free-mode

Hand-guided **free mode** for the Seeed **reBot B601-RS** arm (RobStride motors over CAN), without LeRobot, plus a
**box-limited xylophone music mode**:
- gravity calibration fitted to your own arm
- virtual friction, so the arm stays where you leave it
- kinesthetic demo recording with automatic **replay**, so training images contain no hand
- leader-arm teleoperation
- overhead-camera checks for the xylophone

Built in one day at a hackathon. **Has run on a real B601-RS:** the driver, teleop, free mode, gravity calibration,
virtual friction, and music mode with live tuning. **Mock mode only:** demo replay and the songs.
`python tests/test_mock.py` checks the whole stack without hardware.

## What's in it

| module | what |
|---|---|
| `b601free/b601.py` | follower driver (MIT mode, limits, temperature protection, safe two-stage return to zero), ported from Seeed's LeRobot plugin. Adds per-command step caps, a stiff-plus-gravity mode, a fitted gravity model and a **gripper lock** |
| `b601free/free.py` | **free mode**: gravity feed-forward, virtual friction, automatic calibration, pose teaching, demo record and replay, and **music mode** |
| `b601free/gravity.py` | URDF gravity model (Seeed's, vectorised) plus a **regressor** `g(q) = Y(q)·θ` for fitting link masses and centres of mass |
| `b601free/xylo_vision.py` | overhead-camera bar detection: background consensus, order-robust, moved-check |
| `b601free/record.py` | 30 Hz leader-follower teleop and episode recording (reBot Arm 102 leader) |
| `check.py`, `leader.py`, `cameras.py`, `preview.py`, `keys.py`, `mock.py`, `common.py` | joint-direction check, leader reader, threaded UVC cameras, preview, terminal keys, hardware fakes, helpers |

## Install

```bash
git clone https://github.com/srirangam-r/rebot-b601-free-mode && cd rebot-b601-free-mode
python -m venv .venv && source .venv/bin/activate
pip install -e .            # numpy, opencv-python, motorbridge, motorbridge-smart-servo
```

CAN (PCAN-USB or another SocketCAN adapter), once per boot or replug:
```bash
sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 txqueuelen 1000 && sudo ip link set can0 up
```
The leader is a USB-serial servo bus on `/dev/ttyUSB0`; your user needs to be in the `dialout` group. Everything also
runs without hardware with `--mock`.

## Units
`q`, actions and joint limits are in **follower motor-frame degrees** (Seeed's `math.degrees(state.pos)`). `qvel` is in
deg/s and `tau` in N·m (motor-reported). The URDF takes the same raw angles in radians: joint1..joint6 are pan, lift,
elbow, wrist_flex, wrist_yaw, wrist_roll.

## Gripper lock
The gripper is **locked by default** (`B601Config.gripper_locked`):
- zero torque every tick;
- no close-to-zero check at connect;
- left out of the return to zero;
- the recorded gripper action is its measured position.

This setup had a mallet taped into the gripper. For normal grasping, pass `--unlock_gripper`.

## Free mode

```bash
python -m b601free.free
```
It starts **locked** (HOLD). Hold the arm and press `f`: stiffness ramps to zero over 3 s and the arm floats on gravity
compensation and virtual friction.

| key | action |
|---|---|
| `f` | release (FREE) / lock where it is (HOLD) |
| `a` | **automatic calibration** from the current pose (see below) |
| `c` / `x` | calibration sample by hand at this pose / refit |
| `t` then `0`-`9` | save this pose as **hover** slot N (digits alone never overwrite; `poses.bak.json` keeps the last version) |
| `b` then `1`-`8` | save this pose as the **contact** pose of bar N |
| `v` | overhead camera: store the xylophone bar positions |
| `m` | **music mode** (below) |
| `h` / `z` | move by itself to `--ready` / home. Refused near the xylophone |
| `r` / `s` / `d` | record a demo (from FREE) / stop, replay and save / discard |
| `q` | quit: hold, slow return to zero, torque off. Refused while far from home; press twice to force |

Each session writes a 30 Hz CSV log to `b601free/logs/` (angles, velocities, measured torque, feed-forward).

### Gravity calibration
1. **Measure:** `a` moves the arm itself through 9 poses (lift ±20, elbow ±25, wrist ±30, plus two combinations). `c`
   samples the current pose by hand. At each pose it holds, waits until every joint moves less than 0.4° in 0.4 s, and
   averages the motor torque for 0.5 s.
2. **Fit:** gravity torque is linear in each link's mass and mass × centre of mass, so these are fitted by **ridge
   least squares** on all samples ever taken (`gravity_samples.json`), pulled toward the URDF values. Ordinary links
   may change by about 25%; the tool link gets about ±0.4 kg for a camera or tool payload. The result goes in
   `gravity_fit.json`, used by free mode, replay and the driver's `--gravity_comp`. This is the standard identification
   recipe; see e.g. [arXiv 2405.11114](https://arxiv.org/abs/2405.11114).
3. **Cover your working region:** sample in every region you'll use, including near home.

On the arm this was built on, the fit halved the error (elbow 0.39 → 0.18 N·m mean).

### Virtual friction: it stays where you leave it
These motors have almost no friction, so even 0.2 N·m of leftover model error makes a zero-stiffness arm creep.
Industrial cobots hide this with gearbox friction. Free mode adds it virtually, using the haptics stick-slip proxy
(Hayward & Armstrong):
- each joint is tied to an anchor by a spring with **1.5° of slack**;
- the anchor only slides when you push past the slack;
- so you feel a constant light friction (about 0.65 N·m at lift and elbow), and model error below that can't move it.

Tune with `--vfric_scale` (lower is lighter) and `--vfric_deg`. `--no_vfric` gives a pure float with FACTR-style
friction compensation instead: zero-mean dither while still, gentle Coulomb assist while moving
([arXiv 2502.17432](https://arxiv.org/abs/2502.17432)).

### Recording demos (kinesthetic, with replay)
```bash
python -m b601free.free --out data/task --instructions "a|b" --front <overhead> --wrist <wrist>
```
1. Press `f`, then `r`, guide the motion, and `s`.
2. The arm returns to where the demo started, waits for you to step back, and **replays** your motion with the stiff
   gravity-compensated controller.
3. During the replay it records both cameras. The action is your trajectory `--lead_ms` ahead; `q`, `qvel` and `tau`
   are what the arm actually did.

Episodes are saved as `<out>/{train,val}/ep_*.npz` (front, wrist, q, qvel, tau, action, instruction).

## Music mode (xylophone)

**Safety design.** An earlier agent-written script sent the arm to Cartesian targets computed from URDF IK and a
camera map, and drove it into a laptop. Music mode never goes anywhere a person hasn't shown it:
- **Joint-space box:** made from the taught poses, plus every planned strike pose at the **maximum** of every live
  setting, plus 3°. Every tick it stops and holds if a target or the measured pose leaves the box.
- **Collision stop:** a joint lagging its target by more than 6° (12° on the wrist).
- **Speed stop:** any joint over its music-mode speed limit.
- **Per-bar strikes:** each bar strikes only between its own taught (or blended) hover and contact.
- **Checks before starting:**
  - hover and contact at most 12° apart (a short drop, not a posture change);
  - bars in order across the xylophone;
  - all poses within the joint limits;
  - entry only from near the safe pose or a bar;
  - the xylophone hasn't moved since the scan.
- **Soft gains** during moves (stiff while striking is optional), plus your calibrated gravity model.
- **Space** stops instantly. `w` is a slow preview that lowers only halfway.

**Teach:**
1. `v`: scan the bars, with the arm out of view.
2. For bars 1 and 8, contact first, then hover: rest the mallet on the bar, `b 1`; lift it straight up 1–2 cm,
   `t 1`. Same for bar 8.
3. `t 9`: a safe pose above the xylophone.

Bars 2–7 get hover and contact poses blended from the end bars at the camera-measured bar positions. Teach any bar
by hand to override its blend.

**Play:** guide the mallet near a bar and press `m`, then `w` for the preview.

| key | action |
|---|---|
| `1`-`8` or `a s d f g h j k` | strike that bar |
| `p` / `b` / `l` | songs: Mary Had a Little Lamb / Happy Birthday (F major, B for B-flat) / Moonlight Sonata opening arpeggios (A minor) |
| `<` `>` | song tempo. Songs are automatically slowed just enough for the arm to keep exact time; one-off leaps get extra time |
| `[` `]` | depth 0.70–1.60 (1.00 = taught contact; never more than 3° past it) |
| `-` `=` | strike time 40–400 ms |
| `,` `.` | backswing 0–2.0 × hover-to-contact (capped at 8° and at the joint limits) |
| `;` `'` | travel **peak** speed 20–150 deg/s (acceleration capped at 5× that per second) |
| `n` | stiff or soft gains while striking |
| space / `m` | stop / leave music mode |

Defaults: depth 1.05, 120 ms, no backswing, 60 deg/s, soft. Set them at start-up with `--depth --down_s --windup
--music_speed --strike_stiff`. The camera is only used at `v` and `m`, so it can be unplugged afterwards
(`--overhead ''` runs without it).

## Leader teleop recording
```bash
python -m b601free.check                     # torque off: leader vs follower angles
python -m b601free.record --out data/task --instruction "..." --front <src> --wrist <src>
```
With torque **off**, the follower's position only updates at connect (motorbridge's `request_feedback()` sends nothing
for RobStride motors). With torque on, every command returns fresh feedback.

## Credits and license
MIT (see `LICENSE`). The motor driver, gravity model and URDF are ported from Seeed Studio's MIT-licensed LeRobot
plugins (see `THIRD_PARTY_NOTICES.md`). Motor I/O goes through [motorbridge](https://github.com/motorbridge/motorbridge)
(MIT).
