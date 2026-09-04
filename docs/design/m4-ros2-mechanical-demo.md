# M4 — the ROS 2 Jazzy mechanical demo

The v1.0.0 milestone that moves the *executor* out of Python and onto a real message bus: **one
`ClearanceEnv` as the only physics in the system**, and **K independent ROS 2 nodes — one per
cooperating car — each subscribing to its own state and its own V2V digest and nothing else**,
publishing the drive command that actually moves its car.

This is the second half of [ADR 0005](../adr/0005-phase-migration-gym-first.md)'s split: the thesis used
SUMO for the RL results and Gazebo for a mechanical proof; we use `f1tenth_gym` for the results and
ROS 2 for the mechanical proof — on a stack that is not end-of-life. Status: **accepted** (2026-09-04) — all three forks confirmed (see the end). To be recorded as **ADR 0011**, which supersedes ADR 0005's *"via `f1tenth_gym_ros`
(1–2 cars)"* clause and nothing else in it.

M3 proved decentralized execution with a Python facade (`LocalOnlyView`) and a pipe harness
(`proc_fleet.py`). M4's job is to make the same property hold when the transport is **real** — DDS
serialization, real ordering, real timing, K OS processes that are the *deployment topology* rather
than a test rig — **without letting the transport change the plant**. A second simulator that silently
disagreed with `ClearanceEnv` would poison every number in `m1-clearance-env.md`,
`m3-decentralized-execution.md` and ADRs 0008/0009/0010, so **faithfulness is the primary gate and the
demo is the by-product**.

## Approach — one plant, K deployed car nodes, no end-of-life dependencies

- **Exactly one physics authority.** One `ClearanceEnv` on the pinned `f1tenth_gym v1.0.0`, inside one
  node. No Gazebo, no `gz sim`, no second vehicle model, no re-derived lane geometry, ACC law, reward
  or termination rule. `controllers.py`, `frenet.py`, `scenario.py`, `obs_spec.py` and
  `clearance_eval.write_run` are **imported, never re-implemented**.
- **The bridge does not call `step()`.** A 10 Hz `step()` is too coarse to put ROS inside the actuation
  loop, so `ClearanceEnv.step` is re-expressed over an additive seam (`set_decision` /
  `joint_action_rows` / `substep` / `commit_step`) and the ROS bridge runs *the same loop*, with one
  substitution.
- **Adopt the naming conventions; depend on nothing end-of-life.** We follow the f1tenth topic
  conventions (`/car{j}/odom`, `/car{j}/drive`) because they are what the ecosystem and a real 1/10 car
  already speak — but a *convention is a name, not a dependency*. **Nothing is vendored from
  `f1tenth_gym_ros`**: it is a Foxy-era project, and its Humble branch constructs `F110Env` from
  f1tenth_gym's *dev* branch, whose config surface differs from what `clearance_env.py` passes — so
  adopting any of it risks dragging in a different gym and re-baselining every published table. The
  car's visual model is a handful of RViz markers sized from `ClearanceEnv`'s own
  `params["length"]/["width"]`, and the road is markers generated from `scenario.centerline_xy` — so
  **no scenario constant is written twice** and there is no asset to keep in sync with an unmaintained
  upstream. Every ROS dependency is a package maintained for the target LTS: `rclpy`, `nav_msgs`,
  `ackermann_msgs`, `tf2_ros`, `visualization_msgs`, `rosbag2`, `rviz2`.
- **Two modes, two claims.** *Lockstep* barriers every substep on an integer index and exists **for
  the gate, not for realism**: it removes timing as a variable so the only remaining difference between
  the ROS run and `python -m caatc.clearance_eval` is the wire's precision (fork 2), which is then
  *measured* against a pre-declared tolerance rather than assumed away. *Async* is the honest demo, with command age, drops and
  real-time factor measured and published.
- **The deployed policy is an exported actor, not a pickle.** `ippo-strict.zip` → a ~12 k-parameter
  `.npz` + `.json` run in numpy, so the ROS image carries **no torch and no SB3**. The CTDE actor is
  literally `policy_net(features[..., :ego_dim])` + `action_net` with `net_arch {"pi": [64, 64]}`,
  Tanh, and no `VecNormalize` anywhere — so the export is three matmuls, and the `joint_pad` contract
  disappears from the deployed path entirely.
- **Nothing is retrained.** M4 deploys the existing weights unchanged. If the ROS graph needed
  retraining to work, that is a **finding**, not a task.

### The load-bearing idea

`ClearanceEnv.step` builds a `(num_agents, 2)` array of `[steer, speed]` rows every substep: row 0 from
`ev_control`, rows `1..K` from `coop_lowlevel`, the rest for HARD's occupants. **ROS replaces exactly
rows `1..K`.** The EV row, the occupant rows, the ACC law, the reward, the termination rules and the
metrics stay the identical code object that produced the published numbers.

> M3 was a *reshape, not a rewrite*. M4 is a **row substitution, not a second simulator**: the boundary
> between "what ROS owns" and "what the published numbers describe" is one array slice wide, and
> therefore auditable.

Three consequences, stated up front:

1. **M4 cannot be a performance claim.** M2 and M3 already sit at the oracle ceiling on STRICT. M4's
   evidence is **structural** (gates that fail if the transport changed the plant, or if a car node can
   see something it should not) plus two capability columns headless cannot fill: the cost of the
   **hardware-shaped float32 wire**, and `t_clear` under **real transport latency and loss** (extending
   M3's numpy dropout column into the time domain).
2. **It is the first time the parameter-shared policy is instantiated K times in K address spaces.**
   Training shared one set of weights; deployment loads K independent copies. `proc_fleet.py` is
   promoted from verification tool to production architecture.
3. **The plant sits behind an interface** (`caatc/plant.py`, one implementation in M4), so a 3D plant or
   real hardware later is a plant swap and a separately-gated milestone — not a rewrite.

### What M4 does not claim

**No 3D physics** (planar single-track behind a ROS interface); **no perception** (car nodes receive
exact plant state as proprioception); **no radio physics** (range, i.i.d. loss and constant latency +
jitter — not fading or contention); **no hardware** (no `f1tenth_system`, no VESC, no particle filter).
The EV and side traffic stay **scripted**; the referee is an **oracle** (GJK collisions, privileged
arclength). What it *does* prove: the trained decentralized policy runs **unchanged**, as K independent
ROS 2 nodes over a real bus at real-time rates, on physics **identical** to the published results.

## Interfaces

Two ament packages; colcon artifacts live in a named volume (`caatc-ros-ws`), never in the tree:

```
ros2/src/caatc_msgs/          # ament_cmake (rosidl): CarState, DriveCmd, V2VDigest, Decision, ...
ros2/src/caatc_ros/           # ament_python
  caatc_ros/{clearance_bridge,car_node,v2v_relay,scene_view,ros_gate,bag_replay}.py
  launch/clearance_demo.launch.py     # 1 bridge + 1 relay + K car nodes + view + foxglove
  config/{easy,hard,strict}.yaml
docker/ros.Dockerfile                 # -> caatc-ros  (ros:humble, no torch, no SB3)
```

`run.sh` gains: `ros-build`, `ros-ws-build`, `ros-fingerprint`, `ros-smoke`, `ros-demo`, `ros-gate`,
`ros-replay`, `export-policy`. The ROS image is distro-parametric (`ARG ROS_DISTRO=jazzy`).

**The three-way role split** — labelled in the launch files *and* in bag metadata, so a reader can tell
the deliverable from the scaffolding:

| role | nodes | privileged? | survives a plant swap? |
|---|---|---|---|
| **DEPLOYED** — the deliverable | `/car{j}/agent` ×K | no — allowlisted topics only | **yes, unchanged** |
| **PLANT + REFEREE** | `/clearance_bridge` | owns ground truth, the EV, occupants, reward, outcomes | replaced |
| **RADIO** | `/v2v_relay` | applies range, loss, latency | replaced by real radio |

## Verification — the culture carried over, extended to M4's premise

M1's gate proved **headroom**; M3's proved the 26-vector **sufficient and local**. M4's unproven premise
is *that a real bus can carry the loop without changing the plant, and that locality survives when a
convenience subscription is the natural way to cheat*. `./run.sh ros-gate` reuses
`clearance_smoke.Gate` and exits non-zero on any failure. The 14 checks, with the load-bearing ones
marked:

1. **`check_env_fingerprint`** — f1tenth_gym commit + numpy/numba/llvmlite/scipy/gymnasium/python
   versions, plus a replay of a golden trace recorded on `master`. Prints **TWIN** (bit-identical ⇒ the
   M1–M3 tables carry over verbatim) or **COUSIN** (divergence measured and *published*, and fork 1's
   sidecar is taken). The **only** check allowed to report a divergence instead of failing — and it
   says which it did.
2. `check_graph` — node/topic/type inventory equals the declared contract; rates within ±10%.
3. `check_frame_agreement` — each car's own `(s, d, lane, tangent)` from the exported road is
   bit-identical to the bridge's. Catches exported-road drift for free.
4. **`check_obs_identity`** — `/car{j}/decision.obs` is **bit-identical** to `env.per_agent_obs(j)` at
   the same substep, ≥200 decision instants × 3 presets. *The most important single check*: it proves
   the deployed policy reads the same 26 numbers it was trained on.
5. **`check_lockstep_replay`** (in-image) — replay the recorded rows into a fresh `ClearanceEnv` in the
   same image and require bit-identical per-substep states and an identical `run_episode` record.
   **`check_published_parity`** (cross-image) — `summarize()` equals the published M2/M3 numbers on
   shared seeds. Deliberately two checks of very different difficulty: (5a) must pass unconditionally;
   (5b) is the cross-interpreter one, and check 1 already predicted its verdict.
6. `check_decision_agreement` — the ROS action sequence equals `SharedPolicySquad`'s on the same
   observations; any mismatch must be a near-tie (`|logit margin| < 1e-6`), and the count is printed.
   Not bit-identity, for the reason `dec_smoke` already documents: cross-process BLAS order can flip an
   argmax on a near-tie, and a flaky gate is worse than no gate.
7. **`check_subscription_hygiene`** — each car node's subscription set **equals** the allowlist, and
   `/caatc/ground_truth` has **zero** policy-node subscribers. Ground truth is published *deliberately*
   so absence of evidence becomes evidence of absence — the one thing `LocalOnlyView` structurally
   cannot do.
8. **`check_bag_replay`** — record a `rosbag2`, then replay **only** the allowlisted topics into a fresh
   car node and require the recorded action sequence exactly. A policy with any hidden input cannot
   pass, and it keeps working after every plant swap, including onto hardware. Strictly stronger than
   `proc_fleet`.
9. `check_locality_injection` — an out-of-range peer at 40 m vs 45 m must leave the digest and the
   26-vector bit-identical; a peer at 20 m must change them (the two-sided form `dec_smoke` uses).
10. **`check_headroom_carryover`** — through the full ROS graph on STRICT, `naive` and `speedup` must
    still **fail** while `ippo-strict` succeeds ≥95% with 3.0 yields. If DDS jitter quietly lets
    convoying through, the scenario got easier in transit and the demo proves nothing.
11. `check_constraint_placement` — a car commanding `coop_speed_max` while in the EV's lane must still
    realize ≤ `ev_lane_speed_cap` in the plant. Fails if anyone moves the cap into the car node.
12. `check_peer_sufficiency` — the digest carries everything the 26-vector needs, so the observation is
    reassemblable from broadcasts alone.
13. `check_timing` — measured rates, command age, drops and real-time factor, published not asserted.
14. `check_no_eol_deps` — every ROS dependency resolves from the target LTS's own package
    set, and nothing is vendored from an unmaintained upstream. Fails if a dependency is
    added that the distro does not support.

## Success criteria

- **Faithfulness:** checks 3, 4, 5a on ≥10 shared seeds × {easy, hard, strict} in lockstep. Check 5b
  bit-identical (**TWIN**) or with a **published** divergence table (**COUSIN** + fork 1's sidecar).
- **Decentralization survived the move:** checks 7, 8, 9 on all three presets; K distinct PIDs; and a
  one-container-per-car variant run **once for the record** — exactly as M3 used `proc_fleet` for the
  proof and the in-process squad for speed.
- **The scenario is still the scenario:** check 10 on STRICT.
- **Capability columns headless cannot fill:** the `--wire standard` float32 divergence in
  `t_clear`/outcomes; and `t_clear`/success vs transport latency ∈ {0, 20, 50 ms} and loss
  p ∈ {0, 0.1, 0.3}.
- **Artifacts:** mp4s of `naive`, `speedup` and `ippo-strict` on the *same* graph; a bag; a dashboard
  run overlaying the headless curve.

**Genuine failure, named in advance:** check 4 fails (the deployed policy reads something other than
what it trained on — meaningless however well it drives); check 5a diverges *within the image* with no
documented cause (two physics stacks exist and every number is suspect — the milestone-killer); check 8
fails (a car node has a hidden input — M3's central claim falsified in a deployed setting, and reported
as such); the 26-vector turns out not to be reassemblable from broadcasts (M4 is the wrong milestone
and the right one is observation redesign); `naive` starts succeeding.

**Not failure, pre-committed:** real-time factor < 1 in lockstep; a video no prettier than M2's; a
measured, published cross-interpreter divergence; async degradation (measured, not required); RViz being
sluggish on the integrated GPU.

## Implementation plan (ordered, with stop-gates)

1. **M4.0 — the seam and the twin** (~1 day, no ROS code in (a)). (a) the `caatc/` extractions, each
   with a bit-identity test, then re-run `clearance-smoke`, `dec-smoke`, `gym-test` and the published
   summaries — all must come back **identical**, in its own PR. (b) build `caatc-ros`; run
   `check_env_fingerprint` and the M1 headroom gate *inside it*.
   **Stop-gate 1:** (a) not bit-identical ⇒ the seam is wrong, stop; (b) verdict COUSIN ⇒ take fork 1's
   sidecar **before** writing a node.
2. **M4.1 — one car, lockstep** (~1–1.5 days). Bridge + **one** car node on `local-ideal` (no torch, no
   custom-message risk), rows `2..K` still driven by the env. Checks 2, 3, 4, 5a.
   **Stop-gate 2:** obs bit-identity and in-image replay identity on one car, or the transport is wrong
   and nothing built on top would be trustworthy.
3. **M4.2 — the fleet, the radio, the constraint** (~2 days). K car nodes, `v2v_relay`, occupant
   broadcasts, `caatc_msgs`, the exported policy. Checks 5b, 6, 7, 9, 10, 11, 12.
4. **M4.3 — the evidence and the picture** (~1.5 days). `rosbag2` + check 8; `scene_view`; the Foxglove
   layout; `write_run` → dashboard; the mp4s.
5. **M4.4 — the two measured variants** (~1 day). `--wire standard`; async latency/loss sweeps; check
   13. M4's only new result tables.
6. **M4.5 — the record** (~1 day). This doc, ADR 0011, README/CLAUDE/ROADMAP/`run.sh`.

**Honest cost: 8–10 focused working days** (+1–1.5 if fork 1's sidecar is taken); zero training compute.
The tempting quote is ~5.5 — that is the sum of the parts, and it ignores that this is the repo's
*first* ROS 2 code (colcon in a volume, DDS in Docker, sim time, launch composition) and that the 14
gate checks are ~40% of the work.

## Risks

1. **Cross-interpreter numeric drift.** `rclpy` is compiled against the distro interpreter, so the ROS
   image is Python 3.12 while `caatc-gym` is 3.11; a numba/LLVM or numpy difference could move the last
   bits and break check 5b. *Detected at M4.0(b), before anything is built on it.* Ladder: pin
   numpy/numba/llvmlite/scipy identically and re-check → fork 1's sidecar (restores identity **by
   construction**, and the untimed lockstep barrier cannot notice) → only as the owner's explicit call,
   rebase `caatc-gym` on 3.10 and re-verify every table. Never silently.
2. **The 100 Hz round trip not holding real time.** *Detected by check 13.* Mitigations already in the
   design: only `CarState`/`DriveCmd` cross DDS at 100 Hz (observation building and `coop_lowlevel`
   stay inside the car node); no lidar (our road has no walls); TF at 20 Hz; numpy instead of torch. If
   it still fails: publish at the measured factor with a wall-clock overlay and say so. For
   calibration, the 2020 Gazebo demo ran out of memory on two cars.
3. **Scope creep into "a real robot" or a second plant.** `AckermannDriveStamped`, `/tf` and URDF all
   invite odometry noise, `gz sim` and `f1tenth_system` — each of which changes the dynamics and
   invalidates the published numbers. M4's charter is exactly *the same physics, now over ROS, decided
   per car*. The `Plant` seam is built and deliberately **not** used twice in M4.

## Open forks — CONFIRMED (2026-09-04)

The owner's instruction was explicit: **"I want everything to be adapted on ROS 2 even if this requires
too much rework."** All three forks resolve toward the ROS-native option, and two of them override the
recommendation above. What that costs is stated here rather than discovered later.

1. **Where the physics interpreter lives → one interpreter everywhere: `caatc-gym` is rebased on
   Python 3.12** *(overrides the recommendation, which was to keep 3.11 and measure)*. `rclpy` is built
   against the distro interpreter, so ROS pins 3.10; rather than run two Pythons or a sidecar, the whole
   stack moves to 3.10. **Consequence, accepted:** every published table must be re-verified on the new
   interpreter — the M1 headroom gate, M2's EASY/HARD numbers, M3's STRICT/HARD/EASY numbers and the
   K-transfer and dropout columns. Any drift is **re-baselined and documented**, never silently
   absorbed. This is done *first*, before a line of ROS code, so nothing is built on an unverified base.
2. **The parity wire → standard ROS messages only** *(overrides the recommendation of dual
   publication)*. `nav_msgs/Odometry` and `ackermann_msgs/AckermannDriveStamped` carry the loop; no
   custom float64 twin. This is the idiomatic, hardware-facing choice and it is what a real 1/10 car
   speaks. **Consequence, accepted and important:** every `AckermannDrive` field is float32 and
   `Odometry` carries no steering angle and only a quaternion heading, while `delta` and `theta` are
   consumed every substep — so **bit-identity is no longer attainable**, and M4's faithfulness gates
   become *tolerance* gates:
   - `check_obs_identity` and `check_lockstep_replay` assert agreement within a **declared, published
     tolerance** instead of bit-equality, and the *measured* divergence is reported every run.
   - The tolerance is fixed **before** the runs, derived from the wire (float32 ≈ 6 significant digits
     on state, one quaternion round trip on heading), and a run that needs a looser tolerance than
     declared **fails** rather than having the bar moved.
   - `check_published_parity` therefore reports *outcome* equality (success, collisions, yields) plus a
     divergence table on `t_clear`, not bit-identity. Outcome equality is the claim; the digits are the
     measurement.
   The honest framing shifts from "the published numbers carry over verbatim" to "**the ROS deployment
   reproduces the published outcomes within a measured tolerance, and here is the divergence**" —
   which is the claim a hardware-facing system can actually make.
3. **The policy runtime → exported and run without an ML framework in the robot image.** Read as the
   ROS-native reading of the same instruction: the deployed node is a plain ROS 2 node with numpy, not a
   training stack in a robot container — the artifact a Jetson would run. It is the heavier option
   (export tooling plus a parity test asserting identical argmax on 10 000 real observations), which is
   the trade the instruction accepts. The `.zip` path is kept in `caatc-train` for cross-checking.

*(If fork 3 was meant the other way — keep torch/SB3 inside the ROS image — say so and it is a small
change: the node's policy loader is one class behind one interface.)*

*(Design produced by a 4-proposal judged workflow — angles: port `f1tenth_gym_ros`, a thin bridge over
`ClearanceEnv`, Gazebo mechanical fidelity, a deployment shell toward hardware — then synthesized. The
ranking was near-tied (31 / 30 / 30 / 29.5), and the synthesis takes the **contract** from the winner
and the **engine** from the runner-up.)*
