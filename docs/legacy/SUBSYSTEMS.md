> **Legacy (ROS 1 / Python 2).** This describes the 2020 ROS Kinetic / Gazebo stack that was
> removed from `master` in M1 and preserved at tags `v0.1.0`–`v0.3.0`. See [docs/legacy/README.md](README.md)
> and `git checkout v0.3.0`.

# Subsystems

Mechanism-level notes on each layer, grounded in the source. Paths are relative to
`simulator/racecar-simulator/`.

---

## 1. Simulation environment & robot models

Packages: `racecar_gazebo`, `racecar_description`

**Worlds.** `racecar_gazebo/worlds/threeLanes.sdf` is a 100 m straight road (x = −50…50) over a
black ground plane. Four visual-only white markings (`laneEdge1-4` at y ≈ 0, ±0.525, −1.05) divide
it into three ~0.525 m lanes whose centres (+0.2625, −0.2625, −0.7875) are exactly the spawn
y-values the launch files use. The only *physical* boundaries are two grey `sideWalk` slabs;
barrels, dumpsters, hydrants and jersey barriers are scattered on the shoulders.
`threeLanesCurve.sdf` / `threeLanesSmoothCurve.sdf` bend the road with tessellated polylines.

**Robot.** `racecar_description/urdf/racecar.xacro` builds an Ackermann chassis (10 kg, four
0.05 m-radius wheels, 0.325 m wheelbase, steering ±1.0 rad). `racecar.gazebo` attaches a **Hokuyo**
laser (1081 beams, ±135°, 0.1–10 m, 40 Hz, topic `scan`) and a **camera** (1280×1024, 30 Hz, topic
`camera/image_raw`) plus the `gazebo_ros_control` plugin. `ambulance.xacro` is a byte-for-byte clone
differing only in name and colours (yellow chassis); selected per car via the `r_type` arg.

**Odometry.** `gazebo_odometry.py` is the real localization source: it reads the `robot_name` param,
subscribes to `/gazebo/link_states`, and at 20 Hz republishes ground-truth pose as `Odometry` on
`/vescN/odom` and the `map → racecarN/base_link` TF. AMCL (`racecar_localization`) and gmapping
(`racecar_mapping`) are provided but effectively optional, since Gazebo already supplies a perfect
transform.

**Multi-car scaffolding.** `one_/two_/four_racecars.launch` parameterize N cars into separate
namespaces (`r_name → /racecarN`, `vesc_ns → /vescN`, `tf_prefix`) so identical control /
communication / RL stacks run in parallel without topic or TF clashes.

---

## 2. V2V communication

Package: `racecar_communication`

A lightweight broadcast network so every car knows where the others are without sensing them.

- **`send_id_msg.py`** (one per car, `id_publisherN`): reads its own `/vescN/odom` and
  `move_base/local_costmap/footprint`, derives a discrete lane via `whichLane()` (hard-coded
  threeLanes thresholds), and publishes an `IDStamped` — type, x/y, `lane_num`, footprint polygon,
  velocity, `max_vel`, `max_acc` — to the single **un-namespaced** `/id_msgs` topic. That one topic
  *is* the shared radio channel.
- **`rec_id_msg.py`** (one per car, `id_subscriberN`): drops self-messages, range-gates peers
  against `comm_range` (Euclidean distance), and republishes an aggregated
  `/racecarN/ids_combined` (`IDsCombined`) and `/racecarN/footprints_combined` (`FootprintsCombined`).

**Messages.** `ID` (payload), `IDStamped` (one car's broadcast), `IDsCombined` (`ID[6]`),
`FootprintsCombined` (`Polygon[6]`). The fixed length **6 caps the fleet at six cars**; peers are
addressed by `robot_num-1` array index.

---

## 3. Navigation & the custom communication costmap layer

Packages: `racecar_navigation`, vendored `navigation_/costmap_2d`

Each car runs its own `move_base` in namespace `/racecarN`.

**Carlike TEB planner** (`params/teb_planner_params.yaml`): `TebLocalPlannerROS` configured as an
Ackermann car — `min_turning_radius: 0.9`, `wheelbase: 0.34`, `max_vel_x: 0.4`, no reverse, a
rectangular footprint, and kinematic penalties enforcing the non-holonomic constraint.
`move_base_follower.py` converts TEB's `/cmd_vel` Twist into an `AckermannDriveStamped`.

**The CommunicationLayer plugin** (`navigation_/costmap_2d/plugins/communication_layer.cpp`,
registered in `costmap_plugins.xml`) is the project's key navigation contribution. Each update
cycle it clears the previous footprints, converts each peer polygon from `footprints_combined` to
cells, and stamps them `LETHAL_OBSTACLE` via `setConvexPolygonCost` (padded with a hard-coded 1.5 m
`safety_clearing`), merging into the master costmap with `updateWithOverwrite`. The result: standard
`move_base` + TEB transparently plan around cars they never sensed. **It is disabled by default**
(`enabled: false`); launches pass `use_comm_layer:=true` to turn it on.

**Decision glue** (Python): `threeLanes_current_future_pos.py` classifies odom + planned path into
current/future `Lanes_Info`; `customLocalPlanner.py` diffs them into a discrete `NavAction`
(keep/left/right); `isGoalReached.py` signals goal completion; `simpleGoal.py` sends a single goal.

---

## 4. move_car action subsystem

Package: `racecar_move_car`

The bridge between *deciding* and *actuating*. A decision is a `MoveCar` action goal
(`action_source` 0=nav / 1=RL, `control_action` 0=keep / 1=left / 2=right, `acc`) dispatched to one
of three servers:

- **`laneKeeping_action_server.py`** — lateral: camera → bird's-eye → sliding-window lane fit →
  Stanley steering on `drive_parameters`.
- **`laneKeeping_vel_action_server.py`** — longitudinal: Krauss safe following speed from
  `ids_combined`, published on `move_car/desired_lk_vel`.
- **`laneChange_action_server.py`** — runs a V2V feasibility check (classifies A/B/D neighbours,
  Krauss safe-gap) before generating a Stanley trajectory to the target lane.

**`move_car_action_client.py`** arbitrates two decision sources: `nav_master` (from
`customLocalPlanner`) and `rl_master` (hosts `move_car/RL/RLPolicyActionService`). The first RL
request mutes navigation; a `doneRLflag` hands control back. This is where the RL move-aside policy
becomes real steering and throttle, and where feasibility + duration flow back to the agent as its
reward signal.

---

## 5. Low-level control

Package: `racecar_control`

**Command pipeline:** controllers publish `drive_param{velocity, angle}` on `/drive_parameters` →
`drive_param_to_servo.py` → `ackermann_cmd_mux` → `servo_commands.py`, which converts speed with the
key line `throttle = speed * 20.0` (wheel radius 0.05 m ⇒ 1 m/s ≈ 20 rad/s) and fans out to four
wheel-velocity and two steering-position `ros_control` topics.

**Controllers:**
- **Stanley** (`stanley_controller.py`, and inside `lane_keeping.py`): steering `= err_term + ψ`
  with `err_term = k·atan2(err, ks+V)`, `k=0.005`, `ks=0.1`, clipped to ±0.4189 rad. With such a
  small `k`, the heading term ψ dominates in practice.
- **OpenCV lane keeping** (`lane_keeping.py`): bird's-eye warp + sliding-window polynomial fit → 3
  forward waypoints → Stanley; speed from `/desired_vel`.
- **Krauss car-following** (`krauss_model.py`): picks the nearest in-lane leader from `ids_combined`,
  solves the Krauss safe-velocity quadratic, publishes `/desired_vel` — this drives the ambulance.

---

## 6. Reinforcement learning

Packages: `racecar_clear_ev_route` (agent), `racecar_rl_environments` (environment)

Single-agent **tabular Q-learning** that learns a move-aside policy.

- **State** (`/states` service): `agent_vel`, `agent_lane`, `amb_vel`, `amb_lane`, `rel_amb_y`
  (longitudinal gap `amb_x − agent_x` — the "y" name is a known misnomer). Discretized into a 6-D
  Q-table.
- **Actions:** `{change_left, change_right, acc, no_acc, dec}` → `RLPolicyActionService` request
  (`control_action` 0/1/2 + `acc`).
- **Reward** (step reward, default): shaped around the ambulance *accelerating* during the agent's
  action, normalized by execution time.
- **Update:** `Q(s,a) += α·[r + γ·maxₐ' Q(s',a') − Q(s,a)]`. Config defaults: `α=0.7`, `γ=0.5`,
  `ε=1.0` (min 0.01, decay 1e-4), `max_num_episodes=10`. Q-table saved to `saved_variables/Q_TABLE.npy`.
- **Environment** (`ClearEVRouteBasicEnv`, `racecar_clear_ev_route_basic_env.py`): serves
  `/states`, `/reward`, `/startSim`, `/resetSim`, `/closeSim`; decides `is_RL_activated()`
  (ambulance in `[-24, 9]` window) and `are_we_done()`; renders a fresh Gazebo scenario per episode
  with **jinja2** templates.

> Several files in `racecar_rl_environments/src` (`env_server.py`, `states_server.py`,
> `reward_server.py`, `pub.py`, …) are legacy/scaffold duplicates and are **not** launched — the one
> live environment node is `racecar_clear_ev_route_basic_env.py`. See
> [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the RL caveats (disabled lane-changes, fixed start
> position, 10-episode near-pure-exploration run, jinja2 overwriting launch files in place).
