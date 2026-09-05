# 0011. M4: the ROS 2 Jazzy demo, one simulator plus K car nodes

- **Status:** accepted
- **Date:** 2026-09-04
- **Deciders:** Mohammed El-sayed Ahmed
- **Supersedes:** only the part of [ADR 0005](0005-phase-migration-gym-first.md) that said the demo
  would be built "via `f1tenth_gym_ros` (1–2 cars)". Everything else in ADR 0005 stands.

## The problem

[ADR 0005](0005-phase-migration-gym-first.md) split the work into two: do the learning in the plain
simulator first, then build a ROS 2 demo. This mirrors the thesis, which used SUMO for the learning
results and Gazebo for a mechanical proof. The learning half is done: M1 built `ClearanceEnv`, M2
matched the hand-written ideal with one central PPO policy, and M3 showed that a policy where **each
car acts on its own 26 numbers** matches it too.

M4 is the ROS 2 half. Two facts make the route in ADR 0005 unworkable as written:

1. **`f1tenth_gym_ros` is a ROS 2 Foxy bridge (Foxy is unsupported) that exposes 1 or 2 cars** (an
   "ego" and an "opponent"). Our scenario needs 1 EV plus K=3 cooperators, up to 7 cars with HARD's
   side traffic. Its Humble branch builds the simulator from f1tenth_gym's *dev* branch, whose settings
   differ from what our `clearance_env.py` uses. So **taking their bridge means taking their
   simulator**, which would change every published number. Using an unmaintained bridge also goes
   against [ADR 0002](0002-migrate-to-ros2-humble-python3.md), whose whole point was to get off
   unsupported software.
2. **M3's result is about each car deciding alone.** In ROS terms that means *one node per car*, with
   V2V as real topics, not an ego / opponent split.

The deeper constraint: three milestones of published numbers describe one specific simulator. **A second
simulator that quietly disagreed with `ClearanceEnv` would make all of them worthless.**

## Options

1. **One `ClearanceEnv` as the only physics; K ROS 2 car nodes; follow the f1tenth topic *names*, copy
   nothing.** **Chosen.** ROS replaces exactly rows `1..K` of the per-tick action array. Every ROS
   dependency comes from the target release's own packages.
2. **Port `f1tenth_gym_ros` from Foxy to Humble and extend it to N cars.** Rejected as the base: days
   of work to end up behind a branch that already exists, on a project that would pull in its own
   simulator and change the tables. **Kept from it:** only the topic names (`/{ns}/odom`,
   `/{ns}/drive`) and the idea of a Dockerfile where the ROS release is a parameter. Names and a
   pattern, not code. Since I want nothing end-of-life in this project, **nothing is copied from it**:
   the car and the road are drawn as RViz markers from `ClearanceEnv`'s own numbers and
   `scenario.centerline_xy`, so no file can drift out of sync with an unmaintained upstream.
3. **A second physics engine (`gz sim` Fortress or Gazebo Classic).** Rejected. It would put the
   "maintainable demo" milestone on two dependencies that are about to expire. f1tenth's `mu`, `C_Sf`
   and `C_Sr` are coefficients of a single-track *tyre model* with no equivalent in a rolling-cylinder
   contact model, so a test comparing a URDF with those parameters would give false comfort about the
   exact sideways behaviour it claims to check. And its *predicted* failure (a maximum acceleration of
   about 0.97 g, and the EV's 8 m/s being above the model's own `v_switch` of 7.319 m/s) has a fix,
   rate-limiting the shared command, that would **change M2 and M3's published tables after the fact**.
   A mechanical demo must not change the research results. **Kept from it:** write no scenario constant
   twice; put the simulator behind an interface; record in each run which simulator produced it; and
   re-run the headroom check through the new stack.
4. **ROS 2 as a pure deployment shell toward real hardware.** Adopted as the *framing*, not as a
   substitute: the vehicle interface is the deliverable and the simulator behind it can be swapped
   (`caatc/plant.py`), but M4 still needs a simulator today, and `ClearanceEnv` is the only one whose
   numbers are published.
5. **Feed the EV's adaptive cruise from received V2V instead of ground truth.** Rejected. Limiting it
   by range would change nothing (the ACC law saturates at 7.9 m, far inside the 25 m radio range), but
   **stale** messages would, and a stale gap would change the simulator. The EV stays a scripted row
   owned by the simulator.

## Decision

Build M4 as **one simulator, K deployed car nodes, no end-of-life dependencies**, as laid out in
[`../design/m4-ros2-mechanical-demo.md`](../design/m4-ros2-mechanical-demo.md):

- **One `ClearanceEnv`** on the pinned `f1tenth_gym v1.0.0`. Its `step()` is rebuilt from four smaller
  calls (`set_decision`, `joint_action_rows`, `substep`, `commit_step`) so the ROS bridge can run the
  same loop one physics tick at a time. **ROS replaces exactly rows `1..K`.** The EV row, the side
  traffic, the ACC law, the reward, the end-of-episode rules and the metrics are the same code that
  produced the published numbers.
- **K `/car{j}/agent` nodes.** Each subscribes to an **allow-list** (its own state, its own V2V digest)
  and nothing else, and publishes its own drive command. A `/v2v_relay` node applies range, message
  loss and delay.
- **Two modes.** *Lockstep* (every tick waits for every node, keyed by an integer tick number) exists
  for the **checks**: it removes timing as a variable. *Async* is the honest demo, with the real-time
  factor, command age and drops measured and published.
- **The policy is exported to plain numpy** (`.npz` + `.json`), so the ROS image carries no torch and
  no stable-baselines3.
- **Nothing is retrained.** If the ROS graph needed retraining to work, that is a finding to report,
  not a task to do.
- **14 checks** (`./run.sh ros-gate`, reusing `clearance_smoke.Gate`). The most important: the
  observation each car node reads agrees with `env.per_agent_obs(j)` within a declared tolerance; a
  recorded run replays into a fresh `ClearanceEnv` and gives the same outcome; each car node's
  subscriptions **equal** its allow-list while `/caatc/ground_truth` has **zero** car-node subscribers;
  a `rosbag2` replay of *only* the allow-listed topics reproduces the actions exactly; and the
  **headroom carries over** (`naive` and `speedup` must still fail on STRICT through the full graph).

**Three choices made on 2026-09-04.** I decided to adapt everything to ROS 2 even where that costs
extra work, and to depend on nothing end-of-life. Two of the three go against what I had first planned:

1. **One Python everywhere.** `caatc-gym` moved to **Python 3.12**, the version Jazzy's `rclpy` is
   built against, instead of keeping 3.11 or adding a helper process. Every published table was
   re-verified on 3.12 *first*, before any ROS code. (Result: 16 of 17 rows reproduce exactly; the
   `ctde-easy` model had to be retrained and its numbers were updated in the docs, see
   `m3-decentralized-execution.md`.)
2. **Standard ROS messages only.** `nav_msgs/Odometry` and `ackermann_msgs/AckermannDriveStamped` carry
   the loop. No custom float64 copy of the same data. **This gives up bit-for-bit equality** (the fields
   are float32, `Odometry` has no steering angle, and the heading goes through a quaternion), so M4's
   faithfulness checks compare **within a tolerance** that is declared *before* the runs, and the
   measured difference is printed every run. A run that needs a looser tolerance fails; the bar is not
   moved. The claim becomes "the same outcomes, plus a measured difference table", which is the claim a
   system facing real hardware can actually make.
3. **No machine-learning framework in the robot image.** The policy is exported and run in numpy, with
   a test that it picks the same action as the `.zip` model on 10,000 real observations.

## Consequences

- **Good:** one Python version across the whole stack, and a wire format a real 1/10 car already speaks.
  The deployed node is a plain ROS 2 node with numpy. The "each car decides alone" claim is proven again
  on a *real* bus with real serialisation, ordering and timing, by checks that fail if the transport
  changed the simulator or if a node can see what it should not. The `rosbag2` replay check keeps working
  after any simulator swap, including onto hardware. The published numbers stay valid **by
  construction** (one simulator, one array slice replaced). The stack is supported to **May 2029**. The
  demo can be shown (RViz), and the deliverable, policy nodes plus V2V plus a vehicle interface, is the
  same object a real 1/10 car needs.
- **Cost:** no 3D physics, no perception, no radio physics, no hardware in M4. The EV and side traffic
  stay scripted, and the referee has full knowledge. Lockstep is not real time, by design. We own a
  bridge (a few small messages and six nodes) instead of adopting one. Roughly **8 to 10 focused working
  days**, plus the re-verification of every table on Python 3.12 (done). No training compute unless a
  table has to be re-measured. Choice 2 costs the bit-for-bit claim and replaces it with a measured
  tolerance.
- **Neutral:** ADR 0005's order (learning first, then the demo) stands and is being followed; only its
  bridge clause is replaced. `caatc/plant.py` is built as the hook for a future 3D or hardware
  simulator and deliberately **not** used twice in M4. That would be M5, with its own ADR, checks and
  re-measuring budget.
