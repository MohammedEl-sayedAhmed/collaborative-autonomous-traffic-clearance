# M5: a 3D plant behind the same seam, then one real car

**Decision:** [ADR 0013](../adr/0013-m5-3d-plant-gazebo-harmonic.md). **Builds on:** the M4 seam and the
M4 checks ([`m4-ros2-mechanical-demo.md`](m4-ros2-mechanical-demo.md), ADR 0011).

**Where it stands (2026-09-12):** decided; **M5.0 and M5.1 done**. The four choices are made (an
F1TENTH-style 1/10 car, one at most; lidar plus a map for positioning; Gazebo Harmonic; nothing
retrained). **There is no real car yet** and there may not be one: M5.0 to M5.2 are the deliverable on
their own, and M5.3 runs only if a car arrives. M5.0 measured the plant (above real time, bit-identical).
**M5.1 put Gazebo behind the seam** (`caatc/gazebo_plant.py`, `./run.sh gazebo-smoke`, `./run.sh
gazebo-fleet`): the whole M4 fleet runs on it unchanged with every check passing, the runs are
bit-identical, and the tables gave the first 3D result: on STRICT and EASY the learned policy carries over
(100% success, three yields, `t_clear` 6.50 s against 6.10 s), on HARD it **collides with the side-lane
occupant on every seed** while the hand-written ideal still succeeds. The numbers are in the M5.1 results
section at the end. Next is M5.2: simulated lidar, a map and AMCL on each car.

## What M5 is for

M0 to M4 answered "can the cars learn to clear the path, each on its own, on a real message bus?" with
numbers that are exact to the tick. They were all measured on `f1tenth_gym`, a 2D car model with an
exact pose. Between that and a real car there are three gaps, and M5 measures them one at a time, on the
same checks, with the same car nodes:

| gap | what the 2D plant assumes | what M5 does about it | milestone |
|---|---|---|---|
| physics | a single-track model with a handful of parameters | a 3D car with mass, tyres, a motor and a steering servo, in Gazebo | M5.1 |
| sensing | every car knows its exact pose | each car finds its pose from a simulated lidar and a map, with AMCL | M5.2 |
| the hardware | none | one real car takes the place of one simulated cooperator | M5.3, only if a car is obtained |

M5 does **not** try to make the cars drive better, and it does **not** retrain. If the policy trained on
the 2D model breaks on the 3D plant, that is a result to publish and the start of a later decision.

## The idea: split the referee from the plant

Today `ClearanceEnv` is two things in one class: the **plant** (the physics, `f1tenth_gym`) and the
**referee** (the road and its lanes, the emergency vehicle's cruise law, the speed rule, the reward, the
end-of-episode rules, the metrics). M4 already cut a seam through it (`set_decision`,
`joint_action_rows`, `substep`, `commit_step`). M5 finishes the cut: the referee keeps all of those
calls, and the physics moves behind a small `Plant` interface with two implementations.

```python
class Plant(Protocol):
    """One physics engine. The referee owns everything else."""
    def reset(self, seed: int, starts: np.ndarray) -> np.ndarray:
        """Place the N cars; return their state (the eight numbers per car the records already hold)."""
    def substep(self, rows: np.ndarray) -> np.ndarray:
        """Apply one (steer, speed) row per car for one 10 ms tick; return the new state."""
    def collisions(self) -> np.ndarray:
        """Which cars touched something during the last tick."""
```

- `GymPlant` wraps `f1tenth_gym`. It must be **bit-identical** to today: the golden traces in
  `caatc/tests/golden/` (recorded before the M4 seam, never regenerated) replay through
  `Referee(GymPlant)` exactly, or the refactor is wrong. This is the same proof M4 used for the seam.
- `GazeboPlant` talks to a headless Gazebo Harmonic. Same interface, different numbers.

**What does not change, by construction:** the car nodes (`car_node.py`, the exported numpy policy), the
V2V relay, the gate, the rosbag replay check, the Episode / Odometry / Drive / Decision messages, and the
referee. A car node cannot tell which plant is behind the bridge. That is the point: the rosbag replay
check (M4, check 8) keeps proving that no node has a hidden input, on any plant.

## The Gazebo plant

**The image.** `docker/gazebo.Dockerfile`, from `caatc-ros` (Jazzy), adding `ros-jazzy-ros-gz` (Gazebo
Harmonic from the ROS apt repository, the official pairing), `ros-jazzy-nav2-amcl` and
`ros-jazzy-slam-toolbox` (M5.2), nothing from any other repository. Every package must be supported to
May 2029 or later (ADR 0013).

**The car model.** The 2020 stack's `racecar.xacro`, `macros.xacro` and the meshes at tag `v0.3.0`
(the MIT racecar: chassis, four wheels, two steering hinges, a Hokuyo lidar link) give the geometry and
inertias. They are ported to a modern SDF model with Gazebo's own systems: Ackermann steering for the
drive, the GPU lidar sensor, an IMU, and an odometry publisher for the wheel odometry a VESC would give.
The 2020 plugins (`racecar.gazebo`) are Gazebo Classic and are **not** reused. How the car node's
`AckermannDriveStamped` (a steering angle and a speed) reaches the wheels is settled in the spike: either
Gazebo's Ackermann system with a small shim, or direct joint controllers on the hinges and wheels. Either
way the car node's message is unchanged. The model's parameters are the **real car's**, measured in
M5.3's first step; until then they are the F1TENTH defaults, and the docs say so.

**The world, generated, never drawn by hand.** `caatc/scenario.py` is the only definition of the road
(a 60 m centerline, a gentle sine of 0.5 m amplitude and 40 m wavelength, three lanes of 0.9 m). A small
script writes the Gazebo world from it: a flat ground with the lane lines painted as a texture, and low
walls a little outside the outer lanes so that a lidar has something to see. The referee and the world
therefore use the same numbers, the same principle M4 applied to the observation. The 2020 worlds
(`threeLanes*.sdf`) are reference only.

**Lockstep.** Gazebo runs paused. Each referee tick, `GazeboPlant.substep(rows)` sends every car its
(steer, speed) command, asks the world to step ten 1 ms physics steps through Gazebo's world-control
service, then reads every car's pose and speed. The bridge, the tick numbers, the re-publish rule and
every M4 check stay as they are. The referee's speed rule (clip, and the STRICT cap while a car is in the
EV's lane) is applied before the plant sees the rows, as today.

**Async.** For the demo Gazebo runs free at real time and the bridge samples it every 10 ms, the same
`--async` mode as M4 with the same measurements (check 13).

## Sensing levels and node roles

| level | milestone | where a car's pose comes from | what it measures |
|---|---|---|---|
| L0 | M5.1 | Gazebo's ground truth, published as today's `/car{i}/odom` | the physics gap alone |
| L1 | M5.2 | a `/car{i}/localization` node on the car's side: simulated lidar + IMU + wheel odometry → AMCL against the generated map | the sensing gap, on top of L0 |
| L2 | M5.3 | the real car's own AMCL, on its own computer, on the real track | everything, for one car |

| role | nodes | may hear | replaced by |
|---|---|---|---|
| DEPLOYED | `/car{i}/agent` ×K (unchanged from M4) | own odom, own joint_states, own digest, Episode | nothing |
| DEPLOYED (new in L1) | `/car{i}/localization` | own lidar, own IMU, own wheel odometry, the map | the real car's localization (L2) |
| RADIO | `/v2v_relay` (unchanged) | every car's odom, Episode | a real radio |
| PLANT | Gazebo Harmonic, headless | the K drive commands, through the bridge | the real car, for one car (L2) |
| REFEREE | `/clearance_bridge` with `Referee(GazeboPlant)` | the K drive and decision topics; the plant's ground truth | nothing: it is the scorer |
| GATE | `/ros_gate` | the graph | nothing |

In L1 the gate's allow-list grows by one node per car, and it must show that `/car{i}/localization`
reads nothing from the ground truth either. In L2 the simulated cars still exist in Gazebo; the real car
appears in the same frame because the Gazebo world is generated from the **same map** the real track was
recorded with, and the real car hears the simulated cars over the relay exactly as it hears real ones.

## Checks: the same habit, applied to M5's claim

M5's unproven claim is *that the behaviour measured on the 2D plant survives real physics and real
sensing, and that the parts we deployed did not have to change for it*. `./run.sh gazebo-gate` runs
these and exits with an error on any failure. The most important ones in bold.

1. **`Referee(GymPlant)` replays every golden trace bit for bit.** The refactor changed nothing.
2. `check_env_fingerprint` (M4 check 1) still says IDENTICAL inside the new image.
3. **Repeatability, measured:** the same seed run twice on the Gazebo plant; the largest difference in
   any car's state over the episode is printed, and the tolerance for every later check is declared from
   it *before* the tables are made. If the spread swamps the effects we want to measure, that is a
   finding and the plan changes.
4. **The car nodes are the merged ones:** the node code's hash equals the M4 release, and the rosbag
   replay (M4 check 8) reproduces every decision and drive from the allow-listed topics alone.
5. The gate (M4 check 7) passes with the L1 allow-list; no car-side node hears the ground truth.
6. **Step responses, published:** a steering step and a speed step on one car, on both plants and, in
   M5.3, on the real car. Rise time, overshoot and steady error side by side. Not pass/fail: a table.
7. **Headroom carries over:** `naive` fails and `ideal` succeeds on STRICT on the Gazebo plant. If
   `ideal` fails, the plant or the world is wrong, not the policy.
8. **The learned policy on the 3D plant:** the M4 tables (success, collisions, `t_clear`, yields,
   return) re-measured on L0 and L1, seeds × repeats. Any change from the 2D numbers is the result;
   nothing is required to match.
9. **Localization error, published (L1):** AMCL pose against ground truth per car, mean and worst, and
   how often the estimated lane was wrong. Declared tolerance from check 3 and the lane width.
10. Timing (M4 check 13): real-time factor of the Gazebo plant with K cars and lidars, in lockstep and
    async, on this machine.
11. **The real car (L2):** the same episode with one real cooperator, at least three runs, with the
    referee's outcomes and the real car's own record (its decisions, its drive commands, its AMCL pose)
    saved alongside.

**Real failure, named in advance:** check 1 fails (the refactor changed the physics); the same seed
gives outcomes that differ between two Gazebo runs (nothing can be measured until the source is found);
`ideal` fails on the Gazebo plant (the world or the car model is wrong); a car node had to be edited to
run on the new plant (the deployment claim from M3 and M4 is broken and must be reported).

**Not failure, decided in advance:** the learned policy doing worse on the 3D plant (that is what M5 is
for); a real-time factor below 1 in lockstep; a localization error that changes outcomes (publish it);
the real car needing a slower `lab` preset.

## Implementation plan, in order, with stop points

0. ~~**M5.0, the spike**~~ done (one day). The `caatc-gazebo` image (`./run.sh gazebo-build`); a car
   model from the f1tenth_gym numbers on the 2020 layout with Gazebo's Ackermann system and a GPU lidar
   (`gazebo/models/racecar`); two worlds (`gazebo/worlds/spike1.sdf`, `spike4.sdf`); `gazebo/spike.py`
   steps the paused world 10 ms at a time through the world-control service and measures
   (`./run.sh gazebo-spike --world spike4`). **Stop point passed:** above real time with four lidars,
   bit-identical repeats, no lidar fallback needed. The `Plant` split is in with check 1 green
   (`caatc/plant.py`, `caatc/tests/test_plant.py`; the golden traces and the ROS fingerprint unchanged).
1. ~~**M5.1, L0**~~ done (one day). `GazeboPlant` (`caatc/gazebo_plant.py`), the generated world
   (`caatc/gazebo_world.py`), plants by name (`caatc/plants.py`), the bridge's and the smoke's `--plant
   gazebo`, the headless smoke (`caatc/gazebo_smoke.py`), the step-response tool
   (`caatc/plant_step_response.py`). Checks 2 to 8 and 10 run; the results are at the end.
2. **M5.2, L1** (6 to 8 days). The lidar, IMU and wheel-odometry topics per car; the map generated from
   the same script; `/car{i}/localization` with AMCL; the gate's new allow-list; checks 5 and 9; the
   tables re-measured with sensed poses. Roadmap item 18 lands here.
3. **M5.3, L2, only if a car is obtained** (8 to 10 days, plus hardware time). Measure the car; map the real track; a `lab` preset
   (a shorter road, lower speeds, added as new rows); the real car's bring-up on Jazzy; safety (a speed
   cap in the referee's rule, an e-stop, a person with the remote); one real cooperator in the episode;
   check 11.
4. **M5.4, the record** (1 to 2 days). This doc's results, ADR 0013's outcome section, README, CLAUDE.

**Honest estimate: 12 to 16 focused working days** for M5.0 to M5.2, no training compute; 8 to 10 more
for M5.3 if a car arrives, plus hardware time that no estimate survives. M4 was planned at 8 to 10 days
and went faster; M5 has two things M4 did not, a physics engine we do not control and, maybe, a car that
can break.

## Risks

1. **Lidar rendering speed without a discrete GPU.** Gazebo's GPU lidar renders. Measured in the spike;
   the fallbacks are named there.
2. **Gazebo is not exactly repeatable.** Check 3 turns this into a declared spread. If the spread is
   large, the tables need more repeats, and the exact replay of M4 becomes "replay within the spread".
3. **The Ackermann port.** Gazebo's Ackermann system speaks a different command than a steering angle
   and a speed. Settled in the spike; the car node's message never changes.
4. **The real car is not the F1TENTH default.** Its parameters are measured first in M5.3; until then
   every 3D number carries the label "F1TENTH defaults".
5. **A 60 m road does not fit in a lab.** The `lab` preset shortens it and slows the cars; it is a new
   preset with new rows, so the published EASY / HARD / STRICT numbers are untouched.
6. **The support horizon.** Everything in the image ends in May 2029 together with Jazzy. When the
   stack moves to the next ROS 2 LTS (Lyrical, with Gazebo Jetty, to May 2031), the SDF models and the
   `ros_gz` bridge move with it; nothing in M5 may depend on a Harmonic-only interface.

## Before M5.3: measure and ask

- The car: chassis and wheelbase, mass with battery, steering angle limits, top speed and acceleration
  limits as the VESC is configured, the lidar model (range, field of view, points, rate), the computer
  and its ROS 2 version.
- The track: its size and shape, where the walls are, how the map is recorded.
- The radio: Wi-Fi and the DDS domain for the real car; the relay runs on the laptop.
- Safety: who holds the remote, the e-stop, the speed cap.

## What M5 does not claim

No perception beyond lidar localization; no learned local planner; the emergency vehicle and the side
traffic stay simulated; one real car, not a fleet; no retraining. A camera, a second real car, or
training on the 3D plant would each be a new decision with its own record.

## M5.0 results: the spike

Measured on 2026-09-11 on the development machine (integrated GPU, no discrete graphics card), inside
`caatc-gazebo` (Gazebo Harmonic 8.15, ROS 2 Jazzy), the server headless with EGL rendering, physics at
1 ms steps, the world stepped ten steps at a time from Python, every car driving straight at 2 m/s with a
1081-point lidar at 40 Hz. Three simulated seconds per run, two identical runs per world.

| world | real-time factor | wall time per 10 ms tick | of which the step request | lidar scans in 3 s | two runs |
|---|---:|---:|---:|---:|---|
| one car, one lidar | 1.32 to 1.51 | 6.6 to 7.6 ms | 0.2 ms | 121 | bit-identical |
| four cars, four lidars | 1.02 to 1.55 | 6.5 to 9.8 ms | 0.2 ms | 120 to 121 | bit-identical |

The car reaches the commanded 2.00 m/s and covers 5.54 m in 3 s (the first part is acceleration), so
Gazebo's Ackermann system with the f1tenth_gym numbers drives as expected; the step responses proper are
M5.1's check 6. The stop point is passed: no need for fewer beams or ray-based scans.

Three lessons that go into the Gazebo plant:

- **A blocking request and a subscription callback must not share a Python process.** With the Gazebo
  Python bindings, the request holds Python's lock while the transport thread needs it to deliver a
  callback: 199 of 200 step requests timed out with one subscription active, 0 of 200 without. The
  listener (clock, poses, sensors) is its own process writing into shared memory; the stepper only
  publishes and requests.
- **The per-step sync signal is the world clock topic**, published every physics step. The world
  statistics topic is 10 Hz and made a tick look like 100 ms.
- **Do not set `<topic>` on per-model plugins.** A literal topic is not scoped to the model, so four
  included cars would share one command topic. The defaults (`/model/<name>/cmd_vel`, `/model/<name>/odometry`)
  are per car. And an empty world-control request means "pause: false": a readiness probe must say
  `pause: true` or the world starts running on its own.

## M5.1 results: the Gazebo plant behind the seam

Measured on 2026-09-12, `caatc-gazebo` image, lockstep, ground-truth poses (sensing level L0), the
`ippo-strict` policy exported to numpy, the referee unchanged. Every car in Gazebo is the same
`racecar` model (f1tenth_gym's numbers on the 2020 layout, Gazebo's Ackermann steering with a gain of 30
and the 2D model's 3.2 rad/s steering rate limit); the road, the lane lines and the walls are generated
from `scenario.py`. Raw outputs: `saved_variables/gazebo/`.

**Repeatability (check 3): bit-identical.** The same seed run twice, on each preset, gives the same
state at every decision step to the last bit, and a recorded command sequence replayed into a fresh
server matches every tick (642 of 642). It was not so at first: reading a car's speed from the odometry
topic *on arrival* gave a speed one tick stale in some runs, a spread of up to 0.16 m over an episode.
The plant now waits for the messages *stamped* with the tick's time (the joint-state message of every
step for pose, heading and steering angle; the odometry of every tick for speed). Gazebo's physics was
deterministic all along.

**Headroom (check 7) and the tables (check 8).** Two seeds each; the 2D plant's numbers for the same
policy on the same seeds are in M4's tables (`t_clear` 6.10 s, three yields).

STRICT:

| policy | success | collisions | mean `t_clear` | EV speed | yields | return | plant real-time factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| ideal | 100% | 0% | 6.50 s | 7.48 m/s | 3.0 | 103.0 | 0.88 |
| naive | 0% | 0% | — | 2.47 m/s | 0.0 | 31.9 | 0.88 |
| speedup | 0% | 0% | — | 2.47 m/s | 0.0 | 31.9 | 0.88 |
| ippo-strict (numpy) | 100% | 0% | 6.50 s | 7.48 m/s | 3.0 | 103.0 | 0.88 |

EASY:

| policy | success | collisions | mean `t_clear` | EV speed | yields | return | plant real-time factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| ideal | 100% | 0% | 6.50 s | 7.48 m/s | 3.0 | 103.0 | 0.88 |
| naive | 0% | 0% | — | 2.47 m/s | 0.0 | 31.9 | 0.91 |
| ippo-strict (numpy) | 100% | 0% | 6.50 s | 7.51 m/s | 3.0 | 103.0 | 0.90 |

HARD:

| policy | success | collisions | mean `t_clear` | EV speed | yields | return | plant real-time factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| ideal | 100% | 0% | 6.50 s | 7.48 m/s | 3.0 | 103.0 | 0.60 |
| naive | 0% | 0% | — | 2.47 m/s | 0.0 | 31.9 | 0.62 |
| ippo-strict (numpy) | **0%** | **100%** | — | 7.42 m/s | 3.0 | -56.3 | 0.62 |

What the tables say:

- **The scenario survives the plant.** `naive` and `speedup` still fail, `ideal` still succeeds on every
  preset, and the emergency vehicle clears in 6.50 s instead of 6.10 s: the 3D car is a little slower
  to steer (below), so the yields take longer. The referee never noticed which plant it was scoring.
- **The learned policy carries over on STRICT and EASY** with the same three yields and the same outcome.
- **On HARD it does not.** On both seeds cooperator 1 merges into occupant 5 (the contact log names the
  pair) at 5.2 s and 5.7 s, while `ideal` merges to the free side and succeeds. The policy learned its
  HARD merge on the 2D dynamics with no margin for a car that steers slower and accelerates faster; the
  hand-written rule has that margin. This is M5's first real result, and by ADR 0013 nothing is retrained
  here: fixing it (training on the 3D plant, or with randomised dynamics) is a separate decision.
- The real-time factor in lockstep is 0.88 to 0.91 with four cars and 0.60 with six (HARD has two
  occupants, each with a lidar), including the referee's Python and a 1 ms margin per tick between
  publishing the commands and stepping. Lockstep does not care; the demo would.

**Step responses (check 6), one car alone** (`./run.sh gazebo-step-response`):

| plant | speed rise to 90% of 4 m/s | speed overshoot | speed steady error | steer rise to 90% of 0.3 rad | steer overshoot | steer steady error | yaw rate at 2 m/s, 0.3 rad |
|---|---:|---:|---:|---:|---:|---:|---|
| f1tenth_gym | 0.55 s | 0.0% | 0.000 m/s | 0.10 s | 6.7% | +0.004 rad | 1.78 rad/s (kinematic 1.87) |
| Gazebo | 0.38 s | 0.0% | 0.000 m/s | 0.21 s | 0.8% | +0.002 rad | 1.88 rad/s (kinematic 1.87) |

The Gazebo car reaches speed faster and steers slower; both hold the commanded values exactly. These are
F1TENTH defaults on both sides; the real car's numbers replace them in M5.3 if a car arrives.

**The deployed fleet on the 3D plant** (`./run.sh gazebo-fleet`: the M4 smoke with `--plant gazebo`,
STRICT, seeds 0 and 1, the three car nodes unchanged, the relay, the gate, a rosbag): every check passes.
The nodes' frames and 26 numbers equal the bridge's to the bit (checks 3, 4); every decision is the
reference policy's (0 of 195 differ, check 6); the record replays exactly through a fresh Gazebo server
(642 ticks per episode, check 5a); every car node is subscribed to exactly its four topics and nobody to
the ground truth (check 7); the rosbag of those topics replays into fresh nodes with identical commands
and decisions (1,284 ticks and 130 decisions per car, check 8). Against the headless 2D run of the same
seeds the outcomes are the same and the return differs by 0.8 (102.98 vs 102.14), printed, not required.
A car node cannot tell which plant is behind the bridge, which was the point.

**Lessons for the next plant, kept here:** read the plant's state from stamped messages and wait on the
stamps; never trust arrival order across topics; a strong steering gain and the 2D model's rate limit
make Gazebo's Ackermann system behave like the 2D car; log every contact pair, because "collision" alone
explains nothing.
