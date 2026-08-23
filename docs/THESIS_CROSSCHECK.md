# Thesis ↔ Code Cross-Check Report

**Thesis:** *Autonomous Traffic Clearance for Emergency Vehicles — A Cooperative Reinforcement Learning Approach* (2020)
**Repo:** `collaborative-autonomous-traffic-clearance` (the Gazebo/ROS half only)
**Method:** Per-chapter cross-check findings consolidated and spot-verified against real source under `simulator/racecar-simulator/` and the `docs/` map.

---

## 1. Scope (read this first)

The thesis spans **two simulators**:

| Platform | Purpose | In this repo? |
|---|---|---|
| **SUMO** | Multi-agent RL training + all aggregate **travel-time results** | **No** — unshipped half |
| **Gazebo/ROS** | "Mechanical" AV platform proving practicality | **Yes** — this repo |
| F1/10 hardware | Physical car | No — abandoned (COVID-19); only inherited MIT-racecar packages |

**Every SUMO-scoped claim below is "not-in-repo (SUMO scope)", NOT a defect.** The repo is the Gazebo/ROS deployment; it re-implements the *single-agent* Q-learning problem on Gazebo but contains no SUMO code, no multi-agent traffic population, and no travel-time metric.

A second axis runs through the whole report: **intended-vs-shipped**. The thesis frequently describes the *intended* design (working lane changes, real learning, randomized starts, AMCL/gmapping localization) while the *shipped* default code does something weaker or different. These are flagged explicitly.

---

## 2. Confirmed matches (thesis ✓ code)

The Gazebo-side design is described accurately. Verified against source:

**World & vehicle**
- Three-lane track: `threeLanes.sdf`, 100 m straight (x=-50..50), three ~0.525 m lanes, lane 0 leftmost (positive y), black road / white lines.
- F1/10 Ackermann agent: 0.325 m wheelbase, steering ±1.0 rad; Hokuyo ±2.356 rad (270°) / 0.1–10 m; camera 1280×1024, fov 1.3962634 rad (~1.4); VESC via `servo_commands`/`ros_control`.

**Control**
- **Krauss longitudinal**: `krauss_model.py` gap `= lead_x − x − (1+0.25)·0.58`, safe-velocity quadratic, `des_vel = min(max_vel, v+a·dt, safe_vel)`; low-level wheel controllers pure-**P** (p=1.0/0.5).
- **Stanley lateral**: heading + cross-track, clipped ±0.4189 rad; steering-hinge controllers **PD** (p=10, d=0.5). *(Gain-placement/magnitude caveat in §4.)*
- **Vision lane-keeping**: bird's-eye homography → threshold → sliding-window re-centering → per-lane quadratic `polyfit` → 3 forward waypoints → inverse warp → Stanley.
- **Lane-change feasibility**: `laneChange_action_server.py` A/B/C/D classification, `isSafeWRTA/B/D()`, `isAOrB()` indeterminate branch, `calcMaxSafeVel` worst-case Krauss. The 6-case module unit test is reproducible.

**Navigation & V2V**
- move_base global **navfn/Dijkstra** + local carlike **TEB** (`min_turning_radius 0.9`, no reverse).
- ROS-actions architecture: `MoveCar` action, master + 3 clients + 3 servers (lk / vel / lc); high-level master = `nav_master` + `rl_master` (6 clients), RL preempts nav.
- `ID.msg` = `{type, x_position, y_position, lane_num, footprint(Polygon), velocity, max_vel, max_acc}` — field-for-field match.
- V2V bus: single un-namespaced `/id_msgs`, drop-self, Euclidean range-gate, no relay. Test result 10 m→3 peers, 1.5 m→vehicles 2&3 follows exactly from the `<=` boundary (dists 1.129 / 1.5 / 2.068 m).

**RL formulation** *(verified against config)*
- State: `agent_vel, agent_lane, amb_vel, amb_lane, rel_amb_y` discretized to a Q-table.
- Reward: `m·(emer_curr_spd − amb_last_velocity)+c`, normalized by execution time; `give_final_reward: False` (disabled final reward — matches thesis).
- Update: `Q += lr·(r + γ·maxQ' − Q)`, ε-greedy; `lr 0.7, γ 0.5`.
- Single-agent env: only racecar1 (agent) + racecar2 (ambulance); activation window **[-24, 9]**.
- Gazebo ranges: agent_max_vel **0.5**/acc **0.0167**, amb_max_vel **1**/acc **0.0333**, comm range **24** — exact.
- "Choose from all actions then check feasibility" retry loop; sim-death termination + relaunch (`pingGazebo`, `relaunchGazeboIfDead`, done-code 4).

**Contributions genuinely present**: Gazebo/ROS AV; V2V-for-sensors (`CommunicationLayer` + `ids_combined`-based Krauss/lane-change); single-agent Q-learning agent+env; modular 12-package playground; extendable RL platform.

---

## 3. Intended-vs-shipped gaps (ranked — the heart of the review)

These are places the thesis describes intended behavior the **shipped default code does not do**. All verified in source.

### 3.1 RL lane changes hard-disabled by default ⚠️ *most important*
`single_agent_qlearning.py::execute_action()` (lines 179-183) remaps `change_left→dec` and `change_right→no_acc` when `enable_lane_changes` is `False` (the default; comment *"for testing purposes only"*). The 5-action move-aside space — **the thesis's core cooperation mechanism** — collapses to 3 longitudinal actions. The lane-change *action server* works; the RL agent just never invokes it.

### 3.2 No start randomization / interaction guarantee inert
`genTemplateArgs()` (lines 371-388): `agent_start_x = random.randint(-18,-18)` (always −18); `genRandLanePos()` commented out, so `agent_start_y = EV_start_y = -0.2625` (middle lane). The thesis's randomized, velocity-ordered interaction guarantee is **not active** — every episode is one fixed geometry.

### 3.3 Near-pure-exploration training
Config: `epsilon 1.0, min 0.01, decay_rate 1e-4, max_num_episodes 10`. `ε = min+(max−min)·exp(−decay·episode)` stays **~1.0** across all 10 episodes. A shipped run is essentially random-action throughout and never exploits a learned policy — so the "trained model" the thesis relies on is barely trained in the ROS default.

### 3.4 Custom lane planner subscribes to the wrong topic
`threeLanes_current_future_pos.py` (line 76) subscribes to `move_base/TrajectoryPlannerROS/local_plan`, but the configured planner is **TEB** (`TebLocalPlannerROS/local_plan`). The future-lane callback never fires in a full nav run; the nav-side lane-change trigger rarely activates. The 7-case unit test still passes because positions are fed directly.

### 3.5 Communication costmap layer off by default + inconsistent range-gating
`costmap_common_params.yaml` → `communication_layer: enabled: false`; `racecar_move_base.launch` → `use_comm_layer` default `false` (the two-racecar launch flips it true). Also `footprints_combined` is written for a peer **before** the range check, so out-of-range peers can still be stamped LETHAL — inconsistent with the "limited communication range" premise.

### 3.6 TEB goal tolerance contradicts "stops exactly at the goal"
`teb_planner_params.yaml` → `xy_goal_tolerance: 5` (comment claims 0.7), plus an **x-only** check in `isGoalReached.py`. A ~5 m loose "goal reached" is the opposite of the thesis's tuning claim and is an unmentioned code-level cause of the reported multi-meter goal errors.

### 3.7 Stanley cross-track term negligible (vs Test 1 narrative)
`stanley_controller.py` (lines 77-83): `err_term = k·arctan2(err, ks+v)` with `k=0.005, ks=0.1` — gain **outside** the arctan (thesis writes `arctan(k·e/(ks+v))`), saturating **<0.5°**. The heading term dominates, so the Stanley **Test 1** story of "cross-track error dominating" with large corrective steering cannot occur as shipped.

### 3.8 Shipped ROS reward is the un-fixed "first design"
The thesis credits a redesigned **weighted** reward (EV-accel + agent-action, ordering acc > change-lane > keep > dec) with eliminating oscillation. `calc_reward` in ROS is **EV-acceleration-only**. The corrected reward is a **SUMO-only** artifact.

### 3.9 gmapping / AMCL bypassed by ground-truth odometry
`racecar_mapping` (gmapping) and `racecar_localization` (AMCL — min 500/max 2000 particles, matching the thesis) both ship, but `gazebo_odometry.py` publishes a perfect ground-truth `map→base_link` transform from Gazebo `link_states`. Localization "works" because pose is ground truth, not because AMCL runs; the `laser_scan_matcher` fake-odometry source the thesis describes is not used.

### 3.10 System Integration Episode 1 is unreachable as shipped
Reported: agent lane 0 / EV lane 2, change-left infeasible, change-right moves agent lane 0→1. With lane changes remapped away (§3.1) and both start lanes hardcoded to the middle lane (§3.2), no lane change is dispatched and that start config is impossible. Episode 1 reflects **intended** behavior, not shipped code.

---

## 4. Other discrepancies (documentation / structure)

- **External-Commands-Following & High-Level-Control-Master nodes**: no nodes by those names. Behavior is realized by `actionGenerator.py` (random-action client, acc 0.023 matches) + `move_car_action_client.py` action-source arbitration (0=nav/1=RL, `doneRLflag` handback). Naming/structure mismatch, not behavioral.
- **V2V prose omission**: the *design chapter's* V2V prose lists velocity/position/lane/max-vel/max-acc but omits **type** and the **footprint polygon** that the code broadcasts and the CommunicationLayer/env depend on. (The Methodology chapter's `ID` description *does* list them — a within-thesis inconsistency.)
- **Fleet cap of 6**: `IDsCombined.msg`/`FootprintsCombined.msg` use fixed `[6]` arrays indexed by `robot_num-1`. Not surfaced in the thesis.
- **EV type string**: code uses `'ambulance'` (`type.data=='ambulance'`), thesis says `'EV'`.
- **jinja2 side effect**: per-episode rendering overwrites the version-controlled `empty_world.launch` / `one_racecar_one_ambulance.launch` in place. Not mentioned in the thesis.
- **`move_base_follower.py`** maps TEB `angular.z` straight to `steering_angle` (units mismatch) — a further contributor to goal error.

---

## 5. SUMO-only results (no code here — not defects)

These thesis claims live entirely in the unshipped SUMO half and **cannot be verified or reproduced from this repo**:

- SUMO environment: netgenerate 3-node/2-edge net, TraCI, cell-discretized lanes, integer ranges (EV [0,10], agent [0,5], EV accel [-2,2]), vehicle types `SUMO_KRAUSS` / `Q_LEARNING_SINGLE_AGENT`.
- Multi-agent evaluation (no termination on agent exit, per-vehicle control, vehicle-count / %AV control).
- **All quantitative RL results**: Travel Time vs Lane Busyness (30%: **6.16** steps / 12.3% delay with RL vs **18.56** / 37.1% without; ideal 50 steps = 500 cells @ 10 cells/s); Travel Time vs RL Percentage (monotonic ↓); **50,000-episode** train + 50,000-episode test, EV always optimal.
- The oscillation-fixing weighted reward (§3.8).
- SUMO randomization scheme (randomized agent start + EV lane in training; EV lane only in testing, agent fixed at 149).
- MARL environment + "benchmark problem for MARL" contribution.
- "Preceding vehicles, **each** using a trained RL model" (plural/cooperative) and "two RL platforms on both simulators" — only one platform (Gazebo), **single-agent**, is here.

> Note the contrast: the SUMO side *randomizes* starts to aid generalization; the shipped ROS side does the **opposite** (fixed −18, middle lane) — §3.2.

---

## 6. In thesis, not in this repo

- Entire SUMO half (§5).
- Multi-agent cooperative RL (repo is strictly single-agent: one agent + one ambulance; `/states` models exactly one ambulance).
- Working RL-driven lane changes (§3.1); randomized velocity-ordered starts (§3.2).
- `laser_scan_matcher` fake-odometry feeding gmapping (§3.9).
- Corrected weighted reward (§3.8).
- F1/10 hardware (abandoned; only inherited MIT-racecar packages, hokuyo blacklisted, bag-replay launches broken).
- Separately-named external-commands / high-level-master nodes (realized as arbitration).

## 7. In code, not in thesis

- `enable_lane_changes` param (toggle to restore lane changes over the remap).
- `gazebo_odometry.py` ground-truth transform standing in for LiDAR/ESC + AMCL/gmapping.
- Fixed `[6]` fleet cap; V2V `type` + `footprint` fields (footprint consumed by CommunicationLayer).
- jinja2 overwrites tracked launch files each episode.
- Known bugs/caveats: TEB `xy_goal_tolerance 5`, TrajectoryPlannerROS-vs-TEB topic mismatch, unconditional footprint write vs range-gated IDs, `move_base_follower` steering-units mismatch, lane-keeping crash on empty pixels (now guarded), `single_agent_qlearning.launch` config-load previously commented out.

---

## 8. Bottom line

For the half it actually contains, the thesis describes the code with high fidelity — architecture, controller math, message schemas, and RL formulation nearly all match, several exactly. The dominant caveat is a consistent **intended-vs-shipped drift in the RL agent**: lane changes disabled, starts fixed, ε ~1.0 over 10 episodes — so the showcased cooperative move-aside behavior is effectively **inert by default** and the Gazebo agent barely learns. All headline quantitative results are **SUMO-scope** and unreproducible here. A handful of integration-level defects (bypassed AMCL/gmapping, disabled comm layer, wrong local-plan topic, 5 m goal tolerance) mean the *integrated* system leans on idealized inputs even though the individual modules are real and largely correct.
