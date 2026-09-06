# M4: the ROS 2 Jazzy demo

M4 moves the cars' decision-making out of one Python program and onto a real message bus: **one
`ClearanceEnv` as the only physics in the system**, and **K independent ROS 2 nodes, one per
cooperating car**, each publishing the drive command that actually moves its car. From M4.2 on, a
node subscribes to its own state and its own V2V digest and nothing else; in M4.1 it still hears the
other cars' raw odometry (see the contract).

This is the second half of [ADR 0005](../adr/0005-phase-migration-gym-first.md)'s split: the thesis
used SUMO for the learning results and Gazebo for a mechanical proof; we use `f1tenth_gym` for the
results and ROS 2 for the mechanical proof, on a stack that is not end-of-life. Status: **accepted** on
2026-09-04, all three open questions decided (see the end). Recorded as
[ADR 0011](../adr/0011-m4-ros2-mechanical-demo.md), which replaces only the "via `f1tenth_gym_ros`
(1–2 cars)" part of ADR 0005.

**Where it stands (2026-09-06):** M4.0 and M4.1 are done and reviewed. The images run Python 3.12, all
17 published rows were re-checked, the seam in `ClearanceEnv` is proven to change nothing, and the
`caatc-ros` image is a numerical twin (check 1: **IDENTICAL**). **One car drives over ROS 2 in
lockstep** (`./run.sh ros-smoke`): on strict, easy and hard the ROS run differs from the headless run
on **zero ticks**, the replay is exact, the node's 26 numbers equal the bridge's exactly at every
boundary, its decisions and drive commands are the reference's, and no command was ever stale. M4.2 is
under way: the relay, the gate and the exported policy exist and are checked (see the M4.2 contract at
the end); wiring them into the car node and the smoke is next.

M3 proved that each car can decide alone, using a Python wrapper (`LocalOnlyView`) and a pipe harness
(`proc_fleet.py`). M4's job is to make the same property hold when the transport is **real**: DDS
serialisation, real message ordering, real timing, and K operating-system processes that are the *way
the system is deployed* rather than a test rig, **without letting the transport change the
simulator**. A second simulator that quietly disagreed with `ClearanceEnv` would poison every number in
`m1-clearance-env.md`, `m3-decentralized-execution.md` and ADRs 0008 / 0009 / 0010. So **faithfulness
is the main check, and the demo is the by-product**.

## The approach: one simulator, K deployed car nodes, no end-of-life dependencies

- **Exactly one physics authority.** One `ClearanceEnv` on the pinned `f1tenth_gym v1.0.0`, inside one
  node. No Gazebo, no `gz sim`, no second vehicle model, no second copy of the lane geometry, the ACC
  law, the reward or the end-of-episode rules. `controllers.py`, `frenet.py`, `scenario.py`,
  `obs_spec.py` and `clearance_eval.write_run` are **imported, never rewritten**.
- **The bridge does not call `step()`.** One `step()` is 10 physics ticks with the cars' commands
  decided inside, which is too coarse to put ROS in the loop. So `step()` is now built from four
  smaller calls (`set_decision`, `joint_action_rows`, `substep`, `commit_step`) and the ROS bridge runs
  *the same loop*, with one substitution. **Done**, see the plan.
- **Use the f1tenth topic names; depend on nothing end-of-life.** We follow the f1tenth naming
  (`/car{j}/odom`, `/car{j}/drive`) because that is what the ecosystem and a real 1/10 car already
  speak. But a *name is not a dependency*. **Nothing is copied from `f1tenth_gym_ros`**: it is a
  Foxy-era project (Foxy is unsupported), and its Humble branch builds the simulator from
  f1tenth_gym's *dev* branch, whose settings differ from what `clearance_env.py` uses. Taking any of it
  risks pulling in a different simulator and re-measuring every published table. The car's picture is
  a few RViz markers sized from `ClearanceEnv`'s own `params["length"]` / `["width"]`, and the road is
  markers generated from `scenario.centerline_xy`. So **no scenario constant is written twice**, and
  there is no file to keep in sync with an unmaintained project. Every ROS dependency is a package
  maintained for the target release: `rclpy`, `nav_msgs`, `ackermann_msgs`, `tf2_ros`,
  `visualization_msgs`, `rosbag2`, `rviz2`.
- **Two modes, two claims.** *Lockstep* makes every node wait at every tick (keyed by an integer tick
  number) and exists **for the checks, not for realism**: it removes timing as a variable, so the only
  difference left between the ROS run and `python -m caatc.clearance_eval` is the precision of the
  wire (choice 2 below), which is then *measured* against a tolerance declared in advance instead of
  assumed away. *Async* is the honest demo, with command age, drops and the real-time factor measured
  and published.
- **The deployed policy is an exported network, not a pickle.** `ippo-strict.zip` becomes a
  `.npz` + `.json` pair of about 12k parameters, run in numpy, so the ROS image carries **no torch and
  no stable-baselines3**. The CTDE actor is literally `policy_net(features[..., :ego_dim])` plus
  `action_net`, with `net_arch {"pi": [64, 64]}`, Tanh, and no `VecNormalize` anywhere. So the export
  is three matrix multiplications, and the `joint_pad` trick disappears from the deployed path.
- **Nothing is retrained.** M4 deploys the existing weights unchanged. If the ROS graph needed
  retraining to work, that is a **finding**, not a task.

### The key idea

`ClearanceEnv` builds a `(num_agents, 2)` array of `[steer, speed]` rows every physics tick: row 0 from
`ev_control`, rows `1..K` from `coop_lowlevel`, the rest for HARD's side traffic. **ROS replaces exactly
rows `1..K`.** The EV row, the side-traffic rows, the ACC law, the reward, the end-of-episode rules and
the metrics are the same code that produced the published numbers.

> M3 was a *reshape, not a rewrite*. M4 is a **row substitution, not a second simulator**: the line
> between "what ROS owns" and "what the published numbers describe" is one array slice wide, and so it
> can be audited.

Three consequences, stated up front:

1. **M4 cannot be a performance claim.** M2 and M3 already sit at the ideal on STRICT. M4's evidence is
   **structural** (checks that fail if the transport changed the simulator, or if a car node can see
   something it should not), plus two columns headless runs cannot fill: the cost of the
   **hardware-style float32 wire**, and `t_clear` under **real transport delay and loss** (M3's
   message-loss column, now in the time domain).
2. **It is the first time the shared policy runs as K copies in K separate processes.** Training
   shared one set of weights; deployment loads K independent copies. `proc_fleet.py` goes from
   verification tool to production architecture.
3. **The simulator sits behind a small interface**: the four seam calls on `ClearanceEnv`
   (`set_decision`, `joint_action_rows`, `substep`, `commit_step`). A separate `Plant` protocol with
   a second implementation (a 3D simulator, real hardware) is M5 work with its own checks, not part
   of M4.

### What M4 does not claim

**No 3D physics** (a flat single-track model behind a ROS interface); **no perception** (car nodes
receive the exact simulator state as if it were their own sensors); **no radio physics** (range,
independent random loss, and constant delay plus jitter, not fading or interference); **no hardware**
(no `f1tenth_system`, no VESC, no particle filter). The EV and side traffic stay **scripted**; the
referee has **full knowledge** (exact collisions, exact distance along the road). What it *does* prove:
the trained per-car policy runs **unchanged**, as K independent ROS 2 nodes over a real bus at
real-time rates, on physics **identical** to the published results.

## Interfaces

Two ROS packages; build results live in a named volume (`caatc-ros-ws`), never in the tree:

```
ros2/src/caatc_msgs/          # ament_cmake (rosidl): Episode, Decision, Broadcast, V2VDigest (what no standard message carries)
ros2/src/caatc_ros/           # ament_python
  caatc_ros/{msgs_io,clearance_bridge,car_node,ros_smoke}.py     # M4.1 (done)
  caatc_ros/{v2v_relay,ros_gate}.py                              # M4.2
  caatc_ros/{scene_view,bag_replay}.py, launch/, rviz/           # M4.3
docker/ros.Dockerfile                 # -> caatc-ros  (ros:jazzy, no torch, no SB3; ARG ROS_DISTRO=jazzy)
```

Nodes are started as `python3 -m caatc_ros.<node>` with the image's venv Python; a launch file (M4.3)
will use `ExecuteProcess` with that interpreter. Settings are command-line flags (the bridge also
accepts them as ROS parameters); there is no configuration file.

The car state and the drive command use **standard messages** (`nav_msgs/Odometry`,
`sensor_msgs/JointState`, `ackermann_msgs/AckermannDriveStamped`, choice 2 below). Four small custom
messages carry what nothing standard does: the episode run-control, a car's decision with the
observation it read, and (M4.2) the V2V broadcast and digest.

`run.sh` gains: `ros-build`, `ros-ws-build`, `ros-fingerprint`, `ros-smoke`, `ros-demo`, `ros-gate`,
`ros-replay`, `export-policy`.

**Three roles**, labelled in the launch files *and* in the bag metadata, so a reader can tell the
deliverable from the scaffolding:

| role | nodes | sees everything? | survives a simulator swap? |
|---|---|---|---|
| **DEPLOYED**, the deliverable | `/car{j}/agent` ×K | no, allow-listed topics only | **yes, unchanged** |
| **SIMULATOR + REFEREE** | `/clearance_bridge` | owns the ground truth, the EV, side traffic, reward, outcomes | replaced |
| **RADIO** | `/v2v_relay` | applies range, loss, delay | replaced by a real radio |

## Checks: the same habit again, applied to M4's claim

M1's check proved **headroom**; M3's proved the 26 numbers **enough and local**. M4's unproven claim is
*that a real bus can carry the loop without changing the simulator, and that locality survives when a
convenient extra subscription is the natural way to cheat*. `./run.sh ros-gate` reuses
`clearance_smoke.Gate` and exits with an error on any failure. The 14 checks, with the most important
ones in bold:

1. **`check_env_fingerprint`**: the f1tenth_gym commit plus the numpy / numba / llvmlite / scipy /
   gymnasium / Python versions, plus a replay of the golden traces in `caatc/tests/golden/`. Prints
   **IDENTICAL** (the M1 to M3 tables carry over as they are) or **DIFFERENT** (the difference is
   measured and *published*). The **only** check allowed to report a difference instead of failing,
   and it says which it did.
2. `check_graph`: the set of nodes, topics and message types equals the declared contract, and each
   node reports the interpreter and numpy version it runs on (they must match check 1). In lockstep
   the count is what is checked: exactly one distinct (episode, tick) per topic per tick, re-publishes
   not counted. The "rates within ±10%" clause applies to the async mode only (check 13).
3. `check_frame_agreement`: each car's own `(s, d, lane, tangent)`, carried in its Decision, agrees
   with the bridge's `env.cars[i]` at the same tick: `s`, `d` and `tangent` exact, `lane` equal. In
   M4.1 both sides build the road from the same `scenario.centerline_xy`, so this proves the odometry
   round trip; it becomes a test of an exported road in M4.2. Headings are always compared through
   `wrap_to_pi`.
4. **`check_obs_identity`**: `Decision.obs` from car `i` equals the bridge's snapshot
   `obs_t[i - 1]` (taken at the step boundary, *before* any `set_decision` or `substep`) on every
   element except `heading_err`, which may differ by at most one float32 unit (the heading went
   through a quaternion); over at least 200 decision instants × 3 presets, with the per-element
   difference printed. *The most important single check*: it shows the deployed policy reads the
   same 26 numbers it was trained on. It is a lockstep, zero-loss check by definition.
5. **`check_lockstep_replay`** (inside the image): replay the record into a fresh `ClearanceEnv` in
   the same image and demand **exactly** the same per-tick states and the same episode outcome, bit
   for bit (the replay procedure is in the contract; a divergence surfaces as the tick and the size of
   the first difference, not as a tolerance). **`check_published_parity`** (ROS run vs the headless
   run, same seed): the same success, collisions and yields, `t_clear` within one decision step, and
   the EV's `s` at the same tick within one EV tick; the number of ticks on which the two states
   differ, and the first such tick, are printed. Two checks of very different difficulty on purpose:
   (5a) must pass unconditionally; (5b) is where the float32 wire could show, and check 1 already
   predicts whether the images agree.
6. `check_decision_agreement`: the ROS action sequence equals `SharedPolicySquad`'s on the same
   observations; any mismatch must be a near tie (`|logit margin| < 1e-6`), and the count is printed.
   Not exact equality, for the reason `dec_smoke` already documents: a different order of floating-point
   operations across processes can flip an argmax on a near tie, and a flaky check is worse than none.
7. **`check_subscription_hygiene`**: each car node's set of subscriptions **equals** its allow-list,
   and `/caatc/ground_truth` has **zero** subscribers in any `/car*` namespace. The ground truth is
   published *on purpose*, so "nobody is listening to it" becomes something we can check.
   `LocalOnlyView` could never do that. How it is sampled matters: `ros_gate` is a node that runs for
   the whole episode, waits until every `/car{i}/agent` is visible, then samples
   `get_subscriber_names_and_types_by_node` and `get_subscriptions_info_by_topic` every 100 ms and
   unions the results; car nodes stay alive until the bridge has published `Episode.ENDED` and the
   gate has taken its last sample; the set of node names must equal the declared node set (so no
   helper node can carry an extra subscription); a transient `_NODE_NAME_UNKNOWN_` entry is re-sampled
   and fails only if it persists. The check is per DDS participant, so "K distinct process ids" is what
   covers a fleet composed in one process. **In M4.1 this check tests plumbing, not locality**, because
   the M4.1 allow-list carries every car's raw odometry; locality is tested by check 9 in M4.1 and by
   checks 7 and 8 together from M4.2 on.
8. **`check_bag_replay`**: record a `rosbag2`, then replay **only** the allow-listed topics into a fresh
   car node and demand the recorded action sequence exactly. A policy with any hidden input cannot
   pass, and this keeps working after every simulator swap, including onto hardware. Strictly stronger
   than `proc_fleet`.
9. `check_locality_injection` (pulled forward into M4.1, because it is the one locality check that
   works against a raw-odometry allow-list): feed the node's observation builder odometry with a peer
   at 40 m and at 45 m; the 26 numbers must be identical; a peer at 20 m must change them (the
   two-sided form `dec_smoke` uses). In M4.2 the same check runs on the relay's digest.
10. **`check_headroom_carryover`**: through the full ROS graph on STRICT, `naive` and `speedup` must
    still **fail** while `ippo-strict` succeeds ≥ 95% with 3.0 yields. If DDS timing quietly lets the
    fast convoy through, the scenario got easier in transit and the demo proves nothing.
11. `check_constraint_placement`: a car commanding `coop_speed_max` while in the EV's lane must still
    get at most `ev_lane_speed_cap` in the simulator. Fails if anyone moves the cap into the car node.
    (The seam already enforces this: `substep()` clips and caps rows 1..K.) The record keeps the
    speed before and after the plant's rule and the number of ticks on which the rule changed it;
    that count is this check's evidence.
12. `check_peer_sufficiency`: the digest carries everything the 26 numbers need, so the observation can
    be rebuilt from broadcasts alone.
13. `check_timing`: measured rates, command age, drops and the real-time factor, published, not
    asserted.
14. `check_no_eol_deps`: every ROS dependency comes from the target release's own package set, and
    nothing is copied from an unmaintained project. Fails if a dependency is added that the release
    does not support.

## Success criteria

- **Faithfulness:** checks 3, 4, 5a on at least 10 shared seeds × {easy, hard, strict} in lockstep,
  within the declared tolerance. Check 5b: the same outcomes, with a **published** difference table.
- **Deciding per car survived the move:** checks 7, 8, 9 on all three presets; K distinct process ids;
  and a one-container-per-car variant run **once for the record**, exactly as M3 used `proc_fleet` for
  the proof and the in-process squad for speed.
- **The scenario is still the scenario:** check 10 on STRICT.
- **Columns headless runs cannot fill:** the float32 wire's effect on `t_clear` and outcomes; and
  `t_clear` / success against transport delay ∈ {0, 20, 50 ms} and loss p ∈ {0, 0.1, 0.3}.
- **Artifacts:** mp4s of `naive`, `speedup` and `ippo-strict` on the *same* graph; a bag; a dashboard
  run laid over the headless curve.

**Real failure, named in advance:** check 4 fails (the deployed policy reads something other than what
it trained on, so it is meaningless however well it drives); check 5a differs *inside the image* with
no explained cause (two physics stacks exist and every number is suspect: the milestone killer); check 8
fails (a car node has a hidden input, so M3's central claim is false in a deployed setting, and must
be reported as such); the 26 numbers turn out not to be rebuildable from broadcasts (M4 is the wrong
milestone and the right one is an observation redesign); `naive` starts succeeding.

**Not failure, decided in advance:** a real-time factor below 1 in lockstep; a video no prettier than
M2's; a measured, published cross-image difference; degradation in async mode (measured, not
required); RViz being slow on the integrated GPU.

## Implementation plan (in order, with stop points)

1. **M4.0, the seam and the twin** (about 1 day; no ROS code in (a)).
   **(a) DONE.** `step()` is now the four seam calls run in a loop. Proof that nothing changed: the old
   and new `step()` compared side by side over 2,998 steps (3 presets × 6 seeds × ideal and random
   actions), identical in every observation, reward, flag, info field, target lane and speed, and in
   the inner physics state; six golden traces recorded with the *old* code in `caatc/tests/golden/`
   that a test replays and demands exact equality on; all 79 existing tests plus 18 new ones pass; the
   three gates pass; and **all 17 published rows reproduce**. The seam also enforces the referee's
   rules on rows it did not build (the EV row and side-traffic rows must be untouched; cooperator
   speeds are clipped and, on STRICT, capped).
   **(b) DONE.** `caatc-ros` is built from `ros:jazzy-ros-base` (Python 3.12.3) with the exact package
   versions of `caatc-gym` frozen in as constraints, our package, and `caatc_msgs` built with colcon.
   `./run.sh ros-fingerprint --gate` inside it prints **IDENTICAL** for all six golden traces and
   passes the M1 headroom gate. The 3.12.3 vs 3.12.14 patch-level difference between the two images
   changes nothing.
   **Stop point 1:** passed on both counts.
2. **M4.1, one car, lockstep. DONE.** The groundwork: the observation builder moved out of the env
   into `obs_spec.per_coop_obs` (the two paths agree to the bit), the action constants and the
   decision rule in `caatc/actions.py` so a node never imports the simulator, integer tick stamps, the
   `Episode` / `Decision` messages. The brains as pure Python: `caatc/ros_node_core.py` (the car) and
   `caatc/ros_bridge_core.py` (the plant and referee, the record, the exact replay), tested without a
   bus, wire emulated. The shells: `ros2/src/caatc_ros/caatc_ros/{clearance_bridge,car_node,
   msgs_io,ros_smoke}.py`. **Result** (`./run.sh ros-smoke`, strict / easy / hard, seeds 0 and 1):
   every check passes; the replay is exact (608 to 609 ticks per episode); over 122 (boundary, car)
   pairs every observation element is exact, `heading_err` included; s, d and tangent exact; the ROS
   run and the headless run differ on **0 ticks**; `stale_commands` 0; every decision equals the
   reference policy's and every drive command the reference controller's (check 6, in-image).
   Checks 2 (lite), 3, 4, 5a, 5b, 6 and 9 are in place; `./run.sh ros-smoke --stress` re-publishes
   on every idle spin so the echo path is exercised on the bus too. **Stop point 2: passed.**
   **Stop point 2:** observation agreement and in-image replay agreement on one car, or the transport
   is wrong and nothing built on top could be trusted.
3. **M4.2, the fleet, the radio, the constraint** (about 2 days). K car nodes, `v2v_relay` with the
   `Broadcast` / `V2VDigest` messages, side-traffic broadcasts, the exported policy. Checks 5b, 6, 7,
   10, 11, 12.
4. **M4.3, the evidence and the picture** (about 1.5 days). `rosbag2` and check 8; `scene_view`; the
   RViz layout; `write_run` to the dashboard; the mp4s.
5. **M4.4, the two measured variants** (about 1 day). The float32 wire's effect; the async delay and
   loss sweeps; check 13. M4's only new result tables.
6. **M4.5, the record** (about 1 day). This doc, ADR 0011, README / CLAUDE / ROADMAP / `run.sh`.

**Honest estimate: 8 to 10 focused working days**, no training compute. The tempting estimate is about
5.5, which is the sum of the parts. It ignores that this is the repo's *first* ROS 2 code (colcon in a
volume, DDS in Docker, simulated time, launch files) and that the 14 checks are about 40% of the work.

## Risks

1. **Numeric differences between the two images.** Both images now run Python 3.12, but a different
   numpy, numba or llvmlite build in the ROS image could still move the last digits and break check 5b.
   *Detected at M4.0 (b), before anything is built on it.* The ladder: pin numpy / numba / llvmlite /
   scipy to identical versions in both images and re-check; if that is not enough, publish the measured
   difference (check 1 says DIFFERENT) and carry it through the tolerance checks. Never silently.
2. **The 100 Hz round trip not keeping up with real time.** *Detected by check 13.* Already in the
   design: only the car state and the drive command cross DDS at 100 Hz (building the observation and
   `coop_lowlevel` stay inside the car node); no lidar (our road has no walls); TF at 20 Hz; numpy
   instead of torch. If it still fails: publish at the measured factor with a wall-clock overlay and
   say so. For scale, the 2020 Gazebo demo ran out of memory with two cars.
3. **Scope creep into "a real robot" or a second simulator.** `AckermannDriveStamped`, `/tf` and URDF
   all invite odometry noise, `gz sim` and `f1tenth_system`, and each of those changes the dynamics and
   invalidates the published numbers. M4's charter is exactly *the same physics, now over ROS, decided
   per car*. The seam is the only plant interface in M4; a second implementation is M5.

## The three open questions, decided on 2026-09-04

I decided to adapt everything to ROS 2, even where it costs extra work, and to depend on nothing
end-of-life. All three questions resolve toward the ROS-native option, and two of them go against my
first plan. What that costs is written here rather than found out later.

1. **Where the physics Python lives: one Python everywhere.** `caatc-gym` moved to **Python 3.12**,
   the version Jazzy's `rclpy` is built against. (My first plan was to keep 3.11 and measure the
   difference.) Rather than run two Pythons or a helper process, the whole stack moved. **Cost,
   accepted and paid:** every published table had to be re-checked on the new Python: the M1 headroom
   check, M2's EASY / HARD numbers, M3's STRICT / HARD / EASY numbers and the K and message-loss
   columns. Result: 16 of 17 rows reproduce exactly. The `ctde-easy` model could not be loaded (its
   policy class had been built inside a function, so the file only loaded under the Python that wrote
   it), so it was retrained and its numbers updated in `m3-decentralized-execution.md`. This was done
   *first*, before a line of ROS code.
2. **The wire: standard ROS messages only.** `nav_msgs/Odometry` and `ackermann_msgs/AckermannDriveStamped`
   carry the loop; there is no custom float64 copy of the same data. (My first plan was to publish
   both.) This is the idiomatic, hardware-facing choice, and it is what a real 1/10 car speaks. **Cost,
   accepted and important:** every `AckermannDrive` field is float32, `Odometry` has no steering angle
   and carries the heading as a quaternion, while `delta` and `theta` are used every tick. So
   **bit-for-bit equality is no longer possible**, and M4's faithfulness checks become *tolerance*
   checks:
   - `check_obs_identity` and `check_lockstep_replay` demand agreement within a **declared, published
     tolerance** instead of exact equality, and the *measured* difference is printed every run.
   - The tolerance is fixed **before** the runs, derived from the wire (float32 gives about 6
     significant digits on the state; one quaternion round trip on the heading). A run that needs a
     looser tolerance than declared **fails**; the bar is not moved.
   - `check_published_parity` therefore reports *outcome* equality (success, collisions, yields) plus a
     difference table on `t_clear`, not exact equality. The outcome equality is the claim; the digits
     are the measurement.
   The honest framing moves from "the published numbers carry over as they are" to "**the ROS
   deployment reproduces the published outcomes within a measured tolerance, and here is the
   difference**", which is the claim a system facing real hardware can actually make.
3. **The policy at run time: exported, and run without a machine-learning framework in the robot
   image.** The deployed node is a plain ROS 2 node with numpy, not a training stack inside a robot
   container: the thing a Jetson would run. It is the heavier option (export tooling plus a test that
   the exported network picks the same action as the `.zip` on 10,000 real observations), which is the
   trade I accepted. The `.zip` path stays in `caatc-train` for cross-checking.

## M4.1 contract: topics, messages, the lockstep tick

This is the exact agreement between the bridge and a car node. Both sides are written against it,
and the checks test it. The first draft had an off-by-one between two indices, float time stamps, no
episode identity, double-applied decisions, and a dozen unspecified details. All of them are fixed
below, before any node was written.

### Two indices, two names

The env numbers cars in **agent order**: `i = 0` is the EV, `i = 1..K` are the cooperators, then
HARD's side traffic. Topics, `Decision.car`, `Odometry.child_frame_id` and the rows of the action
array all use `i`. The env's seam and observation calls take the **cooperator index** `j = i - 1`
(`set_decision(j, a)`, `per_agent_obs(j)`, `obs_spec.per_coop_obs(cfg, frame, j, cars)`,
`target_lane[j]`). The contract always writes `i` for the agent index and `j = i - 1` for the
cooperator index. The bridge asserts `1 <= i <= K` on every Decision and Drive it accepts. K=3; in
M4.1 only `i = 1` is driven over ROS, rows 2..3 stay with the simulator.

### Topics

| topic | type | from → to | when |
|---|---|---|---|
| `/caatc/episode` | `caatc_msgs/Episode` | bridge → all | every tick (with the state), and once more with `state = ENDED` |
| `/car{i}/odom` | `nav_msgs/Odometry` | bridge → all, for every car i | every tick |
| `/car{i}/joint_states` | `sensor_msgs/JointState` | bridge → all, for every car i | every tick; `name = ["steering"]`, `position = [delta]` |
| `/car{i}/drive` | `ackermann_msgs/AckermannDriveStamped` | car node i → bridge | every tick |
| `/car{i}/decision` | `caatc_msgs/Decision` | car node i → bridge | every step boundary (`tick % cfg.substeps == 0`) |
| `/clock` | `rosgraph_msgs/Clock` | bridge → tools (rosbag2, RViz, TF) | every tick; **not** for car nodes |
| `/caatc/ground_truth` | `std_msgs/Float64MultiArray` | bridge → nobody | every tick; published **only so check 7 can prove no car node listens to it**; no check reads it |

**Fields.** `Odometry`: `header.stamp` = the tick's stamp (below), `header.frame_id = "map"`,
`child_frame_id = "car{i}"`, `pose.pose.position = (x, y, 0)`, `pose.pose.orientation =
(0, 0, sin(theta/2), cos(theta/2))`, `twist.twist.linear.x = v` (the single-track model's
longitudinal speed); covariances and everything else zero. All float64, so position and speed cross
the wire exactly; only the heading goes through a quaternion, and it comes back **modulo 2 pi** (the
simulator keeps `theta` in `[0, 2 pi)`, the quaternion returns `(-pi, pi]`), which is harmless
because every consumer goes through `wrap_to_pi`. `JointState`: same stamp, `header.frame_id =
"car{i}"`, one joint. `AckermannDriveStamped`: `header.stamp` = the Odometry stamp of the tick the
command is **for**, echoed verbatim; `header.frame_id = "car{i}"`; `drive.steering_angle`,
`drive.speed`; the other three drive fields zero. Both drive fields are **float32** by the message
definition, and the value is rounded on the wire, not at assignment, so a node that logs "what I
sent" must round explicitly: `np.float32(steer), np.float32(speed)`. `Episode`: see the message
file; `start_tick` is where this episode's tick 0 sits on the simulated clock. `Decision`: the bridge
keys it on `(episode, tick)`; `header.stamp` is informational; `car = i`; `obs` is exactly
`np.clip(np.asarray(per_coop_obs(cfg, frame, i - 1, cars), np.float32), -10, 10)` (length
`feature_count(cfg)`), the same array the policy was given; `s, d, lane, tangent` are the node's own
road-frame values at that tick, for check 3. `/caatc/ground_truth`: `data = [stamp_tick, then 8 values
per car in agent order: x, y, wrap_to_pi(theta), v, delta, s, d, lane]`, with `layout.dim` labels
`car` and `field`.

**Stamps are integers.** A tick's stamp is `tick_to_stamp(start_tick + tick, cfg.sim_hz)` from
`caatc/ros_tick.py`: `sec = T // 100`, `nanosec = (T % 100) * 10_000_000` at 100 Hz, never a float
product (`tick * 0.01` mis-encodes 1.7% of ticks, including step boundaries). The first episode starts
at 1.000 s (stamp 0 is "unset" to tf2 and RViz), each later episode one simulated second after the
previous one ended, so simulated time never runs backwards within a bridge process. A node never
builds a stamp: it echoes the Odometry stamp into its Drive and Decision headers and derives the tick
as `stamp_to_tick(sec, nanosec) - episode.start_tick`, which refuses a stamp that is not on a tick.

**Clocks.** Bridge and car nodes run with `use_sim_time = false`. The bridge's re-publish period and
its timeouts use a steady wall clock (`time.monotonic()`); with simulated time they would never fire,
because simulated time only advances when the bridge advances a tick. A car node never subscribes to
`/clock` (with `use_sim_time = true`, rclpy would add that subscription and break check 7). `/clock`
exists for tools only.

**The car node's allow-list (M4.1):** `/caatc/episode`, `/car{i}/odom`, `/car{i}/joint_states` (its
own sensors), `/car0/odom` (the EV's broadcast) and `/car{k}/odom` for every other car k, HARD's side
traffic included. Nothing else. Check 7 demands equality with this set. Range limiting happens inside
the node with the env's own numbers, in the shared `per_coop_obs`. **This allow-list is interim and
wider than M4.2's**: it carries every car's exact position at any distance, so in M4.1 the
decentralised property rests on the node's own code, weaker than M3's `LocalOnlyView`. M4.2 replaces
the raw odometry topics with `/car{i}/v2v`, the relay's range-, loss- and delay-filtered digest, and
only then do checks 7 and 8 test locality rather than plumbing. Check 9 covers locality in M4.1.

**Configuration identity.** Both processes take the same `preset` parameter and build
`caatc.scenario.preset_config(preset)` with defaults (a pure function, so the node can use it too), so `num_agents` (7 on HARD, from `hard_block_sides`, which
`cfg.seed = 12345` fixes), the road, the lanes and every range are identical on both sides. The
bridge's `--seed` is only the `reset` seed and is published in `Episode.seed`. The fallback policy for
simulator-driven cooperators is `LocalIdealCooperator()` with default arguments on the boundary
snapshot `obs_t[j]`, the same policy `dec_smoke` uses as its reference, and the same the M4.1 node
runs. Both sides read these settings from `config/{preset}.yaml`.

**What the node builds from the messages.** A `cars` list of length `num_agents` **indexed by agent
id** (never in arrival order): entry `k` from `/car{k}/odom` with `s, d = frame.project(x, y)`,
`lane = lane_of(cfg, d)`, `v = twist.linear.x`; its own entry additionally with `theta =
quat_to_yaw(orientation)` and `delta = joint_states.position[0]`. `per_coop_obs` reads exactly
`s, d, v, lane, theta, delta` of self and `s, d, v, lane` of every other car; the side flags scan every
other car including the EV (`clear_window` is a geometric window, not a radio gate). Neighbour ties on
`|Δs|` fall back to agent order (a stable sort), which is why the list must be in agent order.

### The lockstep tick

```
bridge                                              car node i  (j = i - 1)
------                                              ----------------------
reset(seed) → tick 0 state; publish Episode(RUNNING,
  start_tick), Odometry×N, JointState×N,
  ground_truth, /clock, all stamped tick 0
                                                    on EVERY callback: latest_stamp[topic] = stamp;
                                                      complete = every required topic carries the newest
                                                      stamp, and Episode for it is known
                                                    if not complete: wait (another callback will re-check)
                                                    tick = stamp_to_tick(stamp) - episode.start_tick
                                                    if (episode, tick) == last handled: re-publish the
                                                      cached Drive (and Decision) ONCE per re-publish
                                                      (each re-publish carries one own-odometry message,
                                                      and each of those earns one answer) and nothing else
                                                    if episode changed or tick == 0: reset target_lane =
                                                      cfg.ev_lane, target_speed = cfg.coop_speed, clear cache
                                                    if tick % cfg.substeps == 0:
                                                        obs = clip(float32(per_coop_obs(cfg, frame, j, cars)))
                                                        a   = policy(obs)
                                                        apply a to own targets (env.set_decision's rules)
                                                        cache and publish Decision(episode, tick, car=i,
                                                          action=a, obs, s, d, lane, tangent)
                                                    row = coop_lowlevel(cfg, frame, s, d, theta, v,
                                                                        target_lane, target_speed)
                                                    cache and publish Drive(stamp echoed, steer, speed)
at the boundary, BEFORE anything else:
  obs_t = env.per_agent_obs_all(); cars_t = env.cars   (the snapshot checks 3 and 4 compare against)
wait for Drive(episode, tick) from every ROS car,
  and Decision(episode, tick) at a boundary;
  first copy wins; a later copy for the same key is
  counted as duplicate_commands and ignored; so is an
  echo of the tick that just finished, if that tick was
  re-published (the node did what the contract asks);
  any other older key is stale_commands and dropped;
  a newer key aborts the run
  re-publish the tick's state every 200 ms (wall clock)
  while waiting, but only when no message arrived in
  the last spin (an answer may be about to complete
  the tick); after a re-publish, once ready, pump for
  20 ms more so the node's echoes land as duplicates
  before the advance; abort after 5 s per tick
  (30 s for tick 0, to cover DDS discovery)
before the first advance of tick 0: exactly one
  publisher on every ROS car's drive topic, or abort
  (a stray node from another run must not answer)
at the boundary, once, from this loop (never from a
  callback): env.set_decision(i - 1, decision[i].action)
  for ROS cars; env.set_decision(j, LocalIdealCooperator()(obs_t[j], cfg))
  for the simulator-driven cooperators
rows = env.joint_action_rows()
rows[i] = (float64(drive.steering_angle), float64(drive.speed))  for ROS cars
done = env.substep(rows)        # the plant clips speeds and applies the STRICT cap
record (see below)
assert (tick % substeps == 0) == (substeps_done was 0)   # the seam and the tick agree on boundaries
if done or env.substeps_done == cfg.substeps: obs, r, terminated, truncated, info = env.commit_step()
  if done and not (terminated or truncated): abort loudly (the seam ended a step early)
if terminated or truncated: publish the final state once more, with Episode.state = ENDED
  (Episode first; a node never answers a tick whose Episode says ENDED); write the record;
  next episode (new seed, start_tick = previous end + 100) or stop
else: tick += 1; publish the new state
```

**On a timeout the bridge aborts**: it writes the partial record with `aborted: true`, the tick and
the missing topics, exits with code 2, and `ros-gate` fails. A signal (Ctrl-C, or the smoke's own
timeout) does the same: the record is written with the reason before the process ends. It **never**
substitutes the simulator's own row for a missing command; that would silently turn a dead node into a
simulator-driven car and every check would still pass. `stale_commands` must be 0 in a healthy
lockstep run; `duplicate_commands` is expected to be non-zero whenever a re-publish happened, and is
not a fault. Every run gets its own **DDS domain** (`ROS_DOMAIN_ID`, chosen by the smoke and written
into the record), so two runs on one machine, or a node left over from an earlier run, cannot hear
each other.

**Who runs the Python.** Every node is started as `python3 -m caatc_ros.<node>` with the image's
venv interpreter (`/opt/venv/bin/python3`, first on `PATH`, `VIRTUAL_ENV` set). `ros2 run` console
scripts and `ros2 launch` `Node(...)` actions would run `/usr/bin/python3`, which has a different
numpy and no simulator; a launch file therefore uses `ExecuteProcess` with the explicit interpreter.
Each node logs `sys.executable` and `numpy.__version__` at start-up, and check 2 fails if they differ
from check 1's fingerprint.

### The record, and the exact replay (check 5a)

Per episode: `preset`, `asdict(cfg)`, `seed`, `episode`, `start_tick`, the list of ROS-driven cars,
the image fingerprint (check 1's versions). Per tick: `rows_applied` (`num_agents × 2` float64, from
`env.rows_applied`), the wire values `(float32 steer, float32 speed)` per ROS car, the speed before
and after the plant's clip/cap rule, `done`, the number of re-publishes. Per boundary: the tick, the
decisions of **all K** cooperators (ROS and fallback), `obs_t` (`K × F` float32), each received
`Decision.obs`, and the node's `s, d, lane, tangent`. Per commit: the 5-tuple (`obs`, `reward`,
`terminated`, `truncated`, `info`). Floats are stored losslessly (`.npz`), never rounded the way the
dashboard writer rounds.

Replay: `env = ClearanceEnv(preset_config(preset)); env.reset(seed)`; at every boundary
`env.set_decision(j, decisions[t][j])` for all j; every tick `env.joint_action_rows()` then
`env.substep(rows_applied[t])` (row 0 and the side-traffic rows are asserted inside `substep`; the
cooperator rows are re-clipped, which is a no-op on already-applied values); `commit_step()` when
`done` or after `cfg.substeps` ticks; compare `env.cars` to the recorded state and each commit 5-tuple
**exactly**. The first tick where the state differs is reported with the size of the difference; a
`ValueError` from `substep` (the recomputed EV row no longer matching) is caught and reported the same
way, so a cross-image difference shows up as a measurement, not as an ownership error.

### Declared tolerances (fixed before any run)

| what | tolerance | why |
|---|---|---|
| observation (check 4), per element | 25 elements **exact**; `heading_err` within one float32 unit (≤ 1.2e-7) | only the heading goes through a quaternion; it is exact to one float64 unit modulo 2 pi and reaches the observation only through `wrap_to_pi(psi - theta)` |
| the steer the plant applied | on the bus: **exactly** the float32 value the node sent (the float64 intent never leaves the node); in the pure-core test with the wire emulated: `|applied − intent|` ≤ `|x| × 6e-8 + 1e-9`, measured | `AckermannDrive.steering_angle` is float32 |
| the speed the plant applied | **exactly** `min(clip(float32(intent), coop_speed_min, coop_speed_max), cap if in the EV lane on STRICT)` | the plant's rule, applied to the value that was on the wire; the number of ticks on which the rule changed the value is printed (check 11's evidence) |
| replay inside the image (check 5a) | **exact** | the recorded `rows_applied` are replayed through `substep()` |
| ROS run vs the headless run, same seed (check 5b) | success, collisions and yields **identical**; `\|Δt_clear\|` ≤ 0.1 s (one decision step); the EV's `s` compared **at the same tick** within 0.08 m (one EV tick at `ev_max_speed / sim_hz`) | speeds cross exactly; the steer's float32 rounding is absorbed by the simulator's bang-bang steering unless it lands within ~6e-8 of the 1e-4 deadband edge, so trajectories are expected to be identical and any divergence is a rare discrete event, reported with its tick |

A run that needs a looser tolerance than this **fails**. The measured values are printed every run.

### What the node is allowed to import

`caatc.actions`, `caatc.scenario`, `caatc.frenet`, `caatc.controllers`, `caatc.obs_spec`,
`caatc.decentralized.LocalIdealCooperator`, `caatc.ros_geometry` and `caatc.ros_tick`. It must **not**
import `caatc.clearance_env`, gymnasium or f1tenth_gym: the node has no simulator. A test
(`test_node_imports.py`) imports the allowed modules in a fresh interpreter and fails if any of the
forbidden ones appears in `sys.modules`.

## M4.2 contract: the fleet, the radio, the learned policy

M4.1 drove one car over ROS 2 while it could still hear every other car's raw odometry. M4.2 drives
all K cars, replaces the raw odometry with a **relay** that models the radio, and runs the **learned**
policy, exported to numpy, instead of the hand-written one. Everything from the M4.1 contract stays;
this section adds to it.

### What changes on the bus

| topic | type | from → to | when |
|---|---|---|---|
| `/car{i}/v2v` | `caatc_msgs/V2VDigest` | relay → car node i, for every cooperator i | every tick, after the relay heard every car's odometry for that tick |

The relay `/v2v_relay` subscribes to `/caatc/episode` and `/car{k}/odom` for every car k (it is the
air: it hears everyone who broadcasts, the EV and the side traffic included). For each cooperator i
and each tick it publishes one `V2VDigest(header.stamp echoed, tick, receiver = i, heard = [...])`
holding one `Broadcast(car, role, x, y, theta, v, tick)` per car k != i whose distance along the road
`|s_k - s_i|` is within the relay's range, after loss and delay (below). The EV is car 0 with role 0;
cooperators role 1; side traffic role 2.

**The car node's allow-list (M4.2):** `/caatc/episode`, `/car{i}/odom`, `/car{i}/joint_states`,
`/car{i}/v2v`. Four topics. It no longer hears any other car directly. From here on checks 7 and 8
test locality, not plumbing.

**Completeness** now means: own odom, own joint_states, the Episode and the digest all carry the
newest stamp. A re-published tick makes the relay re-publish its digests (from its cache, never
recomputed), so the node's rule is unchanged.

### Why the digest is enough (check 12)

The observation builder reads, for every car other than the receiver, only `s, d, v, lane`, and it
uses three gates: the EV block within `v2v_range` (25 m), the neighbour slots within `neighbor_gate`
(= `v2v_range` unless overridden), and the side-lane flags within `clear_window` (6 m). So a digest
that carries every car within `relay_range = max(v2v_range, neighbor_gate, clear_window)` carries
everything the 26 numbers can depend on. The node builds the `cars` list in agent order as before,
and fills the cars it did not hear with a **far-away placeholder** (`s = -1e6`, `d = 0`, `v = 0`,
`lane = 0`). Every gate excludes a placeholder exactly as it excludes the real car that was out of
range, so in lockstep with no loss the node's 26 numbers equal the simulator's to the bit. That is
check 12, and it is a test in the pure cores (`test_ros_core.py`) before it is a check on the bus.
The bridge refuses a configuration where `neighbor_gate > relay_range`.

### Loss and delay (measured in M4.4, wired here)

The relay takes `--loss p` (each `(sender, receiver, tick)` broadcast is dropped independently with
probability p, from a seeded generator whose seed and every drop are written to the relay's record,
so a run can be replayed) and `--delay-ticks d` (a digest at tick t carries the broadcasts of tick
t - d; `Broadcast.tick` says how old each one is; the first d ticks carry nothing). Under loss or
delay the node's observation legitimately differs from the simulator's, so check 4 runs with
`p = 0, d = 0` only; the loss and delay columns are M4.4's measured results, not faithfulness checks.

### The learned policy

Each car node takes `--policy local-ideal` (M4.1's hand-written policy), `--policy numpy:<prefix>`
(an exported actor, `caatc/policies/ippo-strict` by default; loaded with numpy alone; a weights file
that does not match the sha256 in its `.json` is refused), or the two baselines `--policy naive` and
`--policy speedup` for check 10. The exported actor is the deterministic torch policy: on 10,000 real
observations it picks the same action every time (`test_policy_export.py`).

### The checks M4.2 adds

- **6, decision agreement:** every action a car node published equals `NumpyActor.act(obs_t[i - 1])`
  on the bridge's boundary snapshot; a mismatch is allowed only on a near tie (two outputs within
  1e-5), and the count is printed.
- **7, subscription hygiene:** a `ros_gate` node runs for the whole episode, waits until every
  `/car{i}/agent` is visible, samples the graph every 100 ms, and demands: the union of each car node's
  subscriptions equals the four-topic allow-list; no `/car*` node subscribes to `/caatc/ground_truth`
  or to any `/car{k}/odom` with k != i; the set of node names equals the declared set (bridge, relay,
  K agents, gate). Car nodes stay alive until `Episode.ENDED` and the gate's last sample.
- **10, headroom carry-over:** on STRICT through the full graph, `--policy naive` and `--policy
  speedup` fail (0% success) while `numpy:ippo-strict` succeeds on every seed with 3 yields.
- **11, constraint placement:** `--policy speedup` on STRICT: the plant caps the speed on every tick
  the car is in the EV lane; the count of capped ticks is printed and must be > 0.
- **12, peer sufficiency:** check 4 (exact observation agreement) now runs with the digest as the
  node's only source of other cars.
- **5b stays**, now with K ROS cars: the ROS run and the headless run of `SharedPolicySquad` on the
  same seed give the same outcomes, and the EV's `s` at the same tick within one EV tick.

### Records

The relay writes one file per episode too (`relay-<preset>-seed<seed>-ep<n>.json`): its settings and
every drop it made, so a lossy run can be replayed. The bridge's record is unchanged, except that
`ros_cars` now lists all K.

### Where M4.2 stands

Done: the relay's brain and the node-side digest rule (`caatc/ros_v2v.py`), with the in-process proof
that the observation built from a digest equals the simulator's to the bit on all three presets; the
exported numpy actor (`caatc/policy_export.py`, `caatc/policies/`), which picks the same action as the
torch model on all 10,000 real observations tried; the relay node (`v2v_relay.py`: 1,824 digests over
one episode, every one equal to `RelayCore` on the bridge's recorded positions) and the gate node
(`ros_gate.py`, check 7), both checked on a live graph. Still to do: the car node reading its digest
instead of the raw odometry and taking `--policy`; all K cars over ROS; the smoke's fleet mode with the
relay, the gate and checks 6, 7, 10, 11 and 12.

### Node roles, restated

| role | nodes | may hear | replaced by |
|---|---|---|---|
| DEPLOYED | `/car{i}/agent` ×K | own odom, own joint_states, own digest, Episode | nothing: this is the deliverable |
| RADIO | `/v2v_relay` | every car's odom, Episode | a real radio |
| SIMULATOR + REFEREE | `/clearance_bridge` | the K drive and decision topics | a 3D simulator or hardware (M5) |
| GATE | `/ros_gate` | the graph, and the topics it audits | nothing |
