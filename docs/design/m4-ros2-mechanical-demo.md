# M4: the ROS 2 Jazzy demo

M4 moves the cars' decision-making out of one Python program and onto a real message bus: **one
`ClearanceEnv` as the only physics in the system**, and **K independent ROS 2 nodes, one per
cooperating car**, each subscribing to its own state and its own V2V digest and nothing else, and
publishing the drive command that actually moves its car.

This is the second half of [ADR 0005](../adr/0005-phase-migration-gym-first.md)'s split: the thesis
used SUMO for the learning results and Gazebo for a mechanical proof; we use `f1tenth_gym` for the
results and ROS 2 for the mechanical proof, on a stack that is not end-of-life. Status: **accepted** on
2026-09-04, all three open questions decided (see the end). Recorded as
[ADR 0011](../adr/0011-m4-ros2-mechanical-demo.md), which replaces only the "via `f1tenth_gym_ros`
(1–2 cars)" part of ADR 0005.

**Where it stands (2026-09-05):** step M4.0 (a) is done. The images run Python 3.12, all 17 published
rows were re-checked, and the seam in `ClearanceEnv` is built and proven to change nothing (see the
plan below). Next is M4.0 (b), the `caatc-ros` image.

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
3. **The simulator sits behind an interface** (`caatc/plant.py`, one implementation in M4), so a 3D
   simulator or real hardware later is a swap plus its own checked milestone, not a rewrite.

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
ros2/src/caatc_msgs/          # ament_cmake (rosidl): V2VDigest, Decision (the two things with no standard message)
ros2/src/caatc_ros/           # ament_python
  caatc_ros/{clearance_bridge,car_node,v2v_relay,scene_view,ros_gate,bag_replay}.py
  launch/clearance_demo.launch.py     # 1 bridge + 1 relay + K car nodes + view
  config/{easy,hard,strict}.yaml
  rviz/clearance.rviz                 # our own view: car + road markers, nothing copied
docker/ros.Dockerfile                 # -> caatc-ros  (ros:jazzy, no torch, no SB3; ARG ROS_DISTRO=jazzy)
```

The car state and the drive command use **standard messages** (`nav_msgs/Odometry`,
`ackermann_msgs/AckermannDriveStamped`, choice 2 below). Only the V2V digest and the decision need
custom messages, because nothing standard carries them.

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
2. `check_graph`: the nodes, topics and message types match the declared contract; rates within ±10%.
3. `check_frame_agreement`: each car's own `(s, d, lane, tangent)`, computed from the exported road,
   agrees with the bridge's within the declared tolerance. Catches drift in the exported road for free.
4. **`check_obs_identity`**: the observation each car node acts on agrees with `env.per_agent_obs(j)`
   at the same tick, within the declared tolerance, over at least 200 decision instants × 3 presets,
   with the measured difference printed. *The most important single check*: it shows the deployed
   policy reads the same 26 numbers it was trained on.
5. **`check_lockstep_replay`** (inside the image): replay the recorded rows into a fresh `ClearanceEnv`
   in the same image and demand the same per-tick states within tolerance and the same `run_episode`
   outcome. **`check_published_parity`** (across images): `summarize()` gives the published M2 / M3
   outcomes on shared seeds, plus a difference table on `t_clear`. Two checks of very different
   difficulty on purpose: (5a) must pass unconditionally; (5b) is the cross-image one, and check 1
   already predicts its verdict.
6. `check_decision_agreement`: the ROS action sequence equals `SharedPolicySquad`'s on the same
   observations; any mismatch must be a near tie (`|logit margin| < 1e-6`), and the count is printed.
   Not exact equality, for the reason `dec_smoke` already documents: a different order of floating-point
   operations across processes can flip an argmax on a near tie, and a flaky check is worse than none.
7. **`check_subscription_hygiene`**: each car node's set of subscriptions **equals** the allow-list,
   and `/caatc/ground_truth` has **zero** car-node subscribers. The ground truth is published *on
   purpose*, so "nobody is listening to it" becomes something we can check. `LocalOnlyView` could never
   do that.
8. **`check_bag_replay`**: record a `rosbag2`, then replay **only** the allow-listed topics into a fresh
   car node and demand the recorded action sequence exactly. A policy with any hidden input cannot
   pass, and this keeps working after every simulator swap, including onto hardware. Strictly stronger
   than `proc_fleet`.
9. `check_locality_injection`: a car out of range at 40 m and at 45 m must leave the digest and the 26
   numbers unchanged; a car at 20 m must change them (the two-sided form `dec_smoke` uses).
10. **`check_headroom_carryover`**: through the full ROS graph on STRICT, `naive` and `speedup` must
    still **fail** while `ippo-strict` succeeds ≥ 95% with 3.0 yields. If DDS timing quietly lets the
    fast convoy through, the scenario got easier in transit and the demo proves nothing.
11. `check_constraint_placement`: a car commanding `coop_speed_max` while in the EV's lane must still
    get at most `ev_lane_speed_cap` in the simulator. Fails if anyone moves the cap into the car node.
    (The seam already enforces this: `substep()` clips and caps rows 1..K.)
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
   **(b) next:** build `caatc-ros` on `ros:jazzy`; run `check_env_fingerprint` and the M1 headroom
   check *inside it*.
   **Stop point 1:** (a) not identical means the seam is wrong, stop (passed); (b) DIFFERENT means pin
   the numeric libraries to the same versions and re-check **before** writing a node.
2. **M4.1, one car, lockstep** (about 1 to 1.5 days). The bridge plus **one** car node running the
   hand-written local policy (no torch, no custom-message risk), rows `2..K` still driven by the
   simulator. Checks 2, 3, 4, 5a.
   **Stop point 2:** observation agreement and in-image replay agreement on one car, or the transport
   is wrong and nothing built on top could be trusted.
3. **M4.2, the fleet, the radio, the constraint** (about 2 days). K car nodes, `v2v_relay`, side-traffic
   broadcasts, `caatc_msgs`, the exported policy. Checks 5b, 6, 7, 9, 10, 11, 12.
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
   per car*. The `Plant` interface is built and deliberately **not** used twice in M4.

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
