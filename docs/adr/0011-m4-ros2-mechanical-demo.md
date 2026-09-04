# 0011. M4 — the ROS 2 Humble mechanical demo: one plant, K deployed car nodes

- **Status:** accepted
- **Date:** 2026-09-04
- **Deciders:** Mohammed El-sayed Ahmed
- **Supersedes:** the *"via `f1tenth_gym_ros` (1–2 cars)"* clause of
  [ADR 0005](0005-phase-migration-gym-first.md) — and nothing else in it.

## Context and problem statement

[ADR 0005](0005-phase-migration-gym-first.md) phased the migration **gym-first, then a ROS 2 mechanical
demo**, mirroring the thesis's split (SUMO for the RL results, Gazebo for a mechanical proof). The gym
half is done: M1 built `ClearanceEnv`, M2 matched the scripted oracle with centralized PPO, and M3
showed a **decentralized** policy — each car acting on its own 26-feature view — equalling it.

M4 is the ROS 2 half. Two facts make the planned route unworkable as written:

1. **`f1tenth_gym_ros` is a ROS 2 Foxy (EOL) bridge exposing 1–2 agents** (ego + opp), while this
   scenario needs 1 EV + K=3 cooperators (up to 7 cars with HARD's occupants). Its Humble branch
   constructs `F110Env` from f1tenth_gym's *dev* branch, whose config surface differs from what
   `clearance_env.py` passes — so **taking their bridge means taking their gym**, which re-baselines
   every published table. Adopting an unmaintained EOL bridge also cuts against
   [ADR 0002](0002-migrate-to-ros2-humble-python3.md), whose whole point was escaping EOL stacks.
2. **M3's result is about decentralization**, which in ROS terms wants *one node per car* with V2V as
   real topics — not an ego/opp split.

The deeper constraint: three milestones of published numbers describe one specific plant. **A second
simulator that silently disagreed with `ClearanceEnv` would poison all of them.**

## Considered options

1. **One `ClearanceEnv` as the only physics; K ROS 2 car nodes; vendor the f1tenth *contract* but not
   its engine** — **chosen**. ROS replaces exactly rows `1..K` of the per-substep action array.
2. **Port `f1tenth_gym_ros` Foxy→Humble and extend it to N agents** — rejected as the spine: days of
   work to land behind a branch that already exists, on an upstream that would drag in its own gym and
   re-baseline the tables. **Kept from it:** the assets, the topic conventions (`/{ns}/odom`,
   `/{ns}/drive`), the distro-parametric Dockerfile and the Foxglove-first launch pattern, vendored at
   a pinned SHA and never `colcon`-built.
3. **A second physics engine (`gz sim` Fortress or Gazebo Classic)** — rejected: it lands the
   "maintainable demo" milestone on two expiring dependencies; f1tenth's `mu`/`C_Sf`/`C_Sr` are
   single-track *tire-model* coefficients with no analogue in a rolling-cylinder contact, so a
   URDF-vs-params match test would give false assurance about exactly the lateral behaviour it
   measures; and its *predicted* failure (`a_max` ≈ 0.97 g, the EV's 8 m/s past the model's own
   `v_switch` 7.319) has a remedy — rate-limiting the shared command envelope — that **retroactively
   re-prices M2/M3's published tables**. A mechanical demo must not re-price the research milestones.
   **Kept from it:** *no scenario constant is written twice*, the plant-behind-an-interface seam, run
   metadata that distinguishes plants, and re-running the headroom gate through the new stack.
4. **ROS 2 as a pure deployment shell toward hardware** — adopted as *framing*, not as a substitute:
   the vehicle interface is the deliverable and the plant behind it is swappable (`caatc/plant.py`),
   but M4 still needs a plant today, and `ClearanceEnv` is the only one whose numbers are published.
5. **Moving the EV's ACC onto received V2V** — rejected: range restriction is provably inert
   (`acc_target_speed` saturates at 7.9 m, far inside the 25 m gate) but **staleness is not**, and a
   stale lead gap would change the plant. The EV stays a plant-side scripted row.

## Decision

Build M4 as **one plant, K deployed car nodes, a vendored contract**, per
[`../design/m4-ros2-mechanical-demo.md`](../design/m4-ros2-mechanical-demo.md):

- **One `ClearanceEnv`** on the pinned `f1tenth_gym v1.0.0`, re-expressed over an *additive* seam
  (`set_decision` / `joint_action_rows` / `substep` / `commit_step`) so the ROS bridge runs the same
  loop. **ROS substitutes exactly rows `1..K`**; the EV row, the occupants, the ACC law, the reward,
  the termination rules and the metrics stay the identical code object that produced the published
  numbers.
- **K `/car{j}/agent` nodes**, each subscribing to an **allowlist** (its own state, its own V2V digest)
  and nothing else, publishing its own drive command. A `/v2v_relay` node owns range, loss and latency.
- **Two modes:** *lockstep* (an integer-index barrier — an equality test, never a tolerance) exists for
  the **gate**; *async* is the honest demo, with real-time factor, command age and drops published.
- **An exported numpy actor** (`.npz` + `.json`), so the ROS image carries no torch and no SB3.
- **Nothing is retrained.** If the ROS graph needed retraining, that is a finding, not a task.
- **14 gate checks** (`./run.sh ros-gate`, reusing `clearance_smoke.Gate`), the load-bearing ones being:
  the deployed policy's observation is **bit-identical** to `env.per_agent_obs(j)`; a recorded run
  **replays bit-identically** into a fresh env; each car node's subscription set **equals** its
  allowlist while `/caatc/ground_truth` has **zero** policy subscribers; a `rosbag2` replay of *only*
  the allowlisted topics reproduces the actions exactly; and the **headroom carries over** (`naive` and
  `speedup` must still fail on STRICT through the full graph).

**All three forks confirmed (2026-09-04)**, on the owner's explicit instruction to adapt everything to
ROS 2 even at the cost of rework — two of them against the recommendation:

1. **One interpreter everywhere:** `caatc-gym` is rebased on **Python 3.10** (the distro's, which
   `rclpy` is built against) instead of keeping 3.11 or adding a sidecar. Every published table is
   re-verified on the new interpreter *first*, before any ROS code, and any drift is re-baselined and
   documented rather than silently absorbed.
2. **Standard ROS messages only:** `nav_msgs/Odometry` + `ackermann_msgs/AckermannDriveStamped` carry
   the loop, with no custom float64 twin. **This gives up bit-identity** (float32 fields, no steering
   angle in `Odometry`, a quaternion heading round trip) so M4's faithfulness gates become
   **tolerance** gates with the tolerance declared *before* the runs and the divergence published every
   run; a run needing a looser tolerance fails rather than moving the bar. The claim becomes *outcome*
   equality plus a measured divergence table — which is the claim a hardware-facing system can make.
3. **No ML framework in the robot image:** the actor is exported and run in numpy, guarded by a parity
   test against the `.zip` on 10 000 real observations.

## Consequences

- **Positive (as confirmed):** one interpreter across the whole stack, and a wire a real 1/10 car
  already speaks — the deployed node is a plain ROS 2 node with numpy. The decentralization claim is
  re-proven on a *real* bus with real serialization,
  ordering and timing, by gates that fail if the transport changed the plant or if a node can see what
  it should not — and a `rosbag2` replay gate that keeps working after every plant swap, including onto
  hardware. The published numbers stay valid **by construction** (one plant, one array slice of
  substitution). The stack is LTS-supported to 2027, not EOL. The demo is showable (Foxglove/RViz) and
  the deliverable — policy nodes + V2V + a vehicle interface — is the same object a real 1/10 car needs.
- **Negative / trade-offs:** no 3D physics, no perception, no radio physics, no hardware in M4; the EV
  and side traffic stay scripted and the referee is an oracle. Lockstep is not real-time by design. We
  own a bridge (~6 small messages and six nodes) rather than adopting one, and a cross-interpreter
  numeric divergence is possible (detected before anything is built on it, with a documented ladder).
  Honest cost: **8–10 focused working days** plus the re-verification of every published table on
  Python 3.10 (fork 1) — zero training compute unless a table has to be re-baselined. Fork 2 costs the
  bit-identity claim, replacing it with a measured tolerance.
- **Neutral:** ADR 0005's phase decision stands and is being honoured; only its bridge clause is
  superseded. `caatc/plant.py` is built as the seam for a future 3D or hardware plant and deliberately
  **not** used twice in M4 — that is M5, with its own ADR, gates and re-baselining budget.
