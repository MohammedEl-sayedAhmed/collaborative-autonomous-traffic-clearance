# M5: a 3D plant behind the same seam, then one real car

**Decision:** [ADR 0013](../adr/0013-m5-3d-plant-gazebo-harmonic.md). **Builds on:** the M4 seam and the
M4 checks ([`m4-ros2-mechanical-demo.md`](m4-ros2-mechanical-demo.md), ADR 0011).

**Where it stands (2026-09-11):** decided, not started. The four choices are made (an F1TENTH-style
1/10 car, one at most; lidar plus a map for positioning; Gazebo Harmonic; nothing retrained). **There is
no real car yet** and there may not be one: M5.0 to M5.2 are the deliverable on their own, and M5.3 runs
only if a car arrives. The first stop point is the M5.0 spike below, which has to answer two questions
before anything else is built: how fast Gazebo runs headless with lidar on this machine, and how
repeatable it is.

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

0. **M5.0, the spike** (3 to 4 days). The `caatc-gazebo` image; one ported car in an empty world with
   Ackermann drive and a GPU lidar, headless; step it from Python through the world-control service;
   measure the real-time factor with one and with four lidars on the integrated GPU; run the same seed
   twice and print the spread. **Stop point:** if lidar rendering is far below real time, decide
   between fewer beams, a lower lidar rate, or ray-based scans without rendering, and record it. In
   parallel, the `Plant` split with check 1 green.
1. **M5.1, L0** (3 to 4 days). `GazeboPlant`, the generated world, the bridge's `--plant gazebo`. Checks
   2, 4, 5, 6, 7, 8, 10. First tables.
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
