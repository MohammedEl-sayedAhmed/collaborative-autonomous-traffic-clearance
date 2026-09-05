> **Legacy (ROS 1 / Python 2).** This describes the 2020 ROS Kinetic / Gazebo stack that was
> removed from `master` in M1 and kept at tags `v0.1.0` to `v0.3.0`. See [docs/legacy/README.md](README.md)
> and `git checkout v0.3.0`.

# Known issues and fixes

What I found by reading all the code and running every demo in the container (2026). Each entry says
whether it is fixed on this branch, is expected behaviour, or is a real bug left as it was (with the
one-line fix, so you can decide).

---

## Fixed on this branch

### RL launch config <a id="rl-launch-config"></a>
`racecar_clear_ev_route/launch/single_agent_qlearning.launch` had its `<rosparam>` config-load
commented out, so the master aborted immediately with `KeyError: 'max_num_episodes'`. The config
files exist and define the needed params. **Fix applied:** uncommented the two `<rosparam>` lines so
the launch loads `single_agent_qlearning_master_config.yaml` and
`racecar_clear_ev_route_basic_env_config.yaml`. (These are also loaded by `env.launch`; loading them
here too is harmless and makes the launch usable on its own.) A full training run still needs the
environment node from `env.launch` — see [RUNNING.md](RUNNING.md#running-the-rl-scenario).

---

## Environmental / expected (not code bugs)

### `keyboard_teleop` dies in headless runs
`keyboard_teleop{1..4}.py` reads the terminal with `termios`, so it exits with
`termios.error: Inappropriate ioctl for device` when launched headless or backgrounded. Harmless —
the simulation runs fine; teleop only works when you launch interactively. Command a car
programmatically otherwise (see [RUNNING.md](RUNNING.md#driving-a-car)).

### `gzserver` Ogre `setDepthBufferFor` assertion under software GL
Gazebo aborts (exit 134 / SIGABRT) with an Ogre depth-buffer assertion **only when rendering the
car's camera under a software GL renderer** (Xvfb / `LIBGL_ALWAYS_SOFTWARE=1`). With hardware GL
(the default `run.sh` path: `/dev/dri` + your `DISPLAY`) the camera demos (`movecar`, `ev`) run
fine — verified. This is a headless-rendering limitation, not a repo defect.

### Gazebo model database
The worlds reference `model://` URIs from the retired online Gazebo database. The Docker image bakes
those models in and blanks `GAZEBO_MODEL_DATABASE_URI`, so first launch doesn't stall. If you add a
world with new models, add them to `docker/` or `GAZEBO_MODEL_PATH`.

---

## Real bugs left unapplied (behaviour-changing — review before applying)

These are real bugs, but fixing them changes how a research pipeline behaves, and that pipeline
cannot be fully tested without a screen (the camera path needs a GPU and someone driving). Listed
with the fix so you can apply them on purpose.

### TEB local-plan topic mismatch
`racecar_navigation/scripts/threeLanes_current_future_pos.py` subscribes to
`move_base/TrajectoryPlannerROS/local_plan`, but the configured planner is **TEB**, which publishes
`move_base/TebLocalPlannerROS/local_plan`. The future-lane callback likely never fires, so
`customLocalPlanner` rarely sees a real lane-change trigger.
**Fix:** change the subscribed topic to `.../TebLocalPlannerROS/local_plan`.

### RL lane changes are hard-disabled
`single_agent_qlearning.py::execute_action()` unconditionally remaps `change_left → dec` and
`change_right → no_acc` (comment: *"FIXME this is for testing purposes onlyyyyyyyy"*), so despite a
5-action space only the 3 longitudinal actions ever reach `move_car`.
**Fix:** remove the remap so lane-change actions dispatch normally.

### RL start position not randomized
`genTemplateArgs()` sets `agent_start_x = random.randint(-18, -18)` (both bounds equal ⇒ always −18),
and `genRandLanePos()` is commented out, so both agent and ambulance always start in the middle
lane — nullifying lane changes.
**Fix:** widen the range (comment intends −18…12) and re-enable lane randomization.

### `move_base_follower` steering units
`move_base_follower.py` maps TEB's `/cmd_vel.angular.z` (an angular *velocity*, since
`cmd_angle_instead_rotvel: False`) directly to `AckermannDriveStamped.steering_angle` — a units
mismatch.
**Fix:** set `cmd_angle_instead_rotvel: True` in `teb_planner_params.yaml`, or convert ω→steering.

### `xy_goal_tolerance` is 5 m
`teb_planner_params.yaml` sets `xy_goal_tolerance: 5` (the comment claims 0.7). Combined with
`isGoalReached.py`'s x-only check, "goal reached" is very loose.
**Fix:** set it to ~0.2–0.7 m.

---

## Broken launches (out of scope — hardware/legacy demos)

- **`racecar_control/launch/gazebo_sim_joy.launch`** includes `racecar_gazebo/launch/racecar_tunnel.launch`,
  which doesn't exist in this repo (it was a baseline-only world). It's a joystick-in-tunnel demo;
  not needed for the project. To revive it, point the include at `one_racecar.launch` and add a
  joystick.
- **`system/racecar/.../replay_bag_mapping.launch`, `replay_bag_with_lidar_processing.launch`**
  reference a missing `racecar-v1/static_transforms.launch.xml` and the `laser_scan_matcher` package
  (physical-car bag replay). Not part of the simulation pipeline.
- **`hokuyo_node`** is a hardware-only lidar driver and is blacklisted from the build
  (`CATKIN_BLACKLIST_PACKAGES=hokuyo_node` in `run.sh`).

---

## Design constraints & sharp edges

- **Fleet capped at 6 cars** — `IDsCombined`/`FootprintsCombined` are fixed `[6]` arrays indexed by
  `robot_num-1`.
- **Communication layer is off by default** (`enabled: false`); pass `use_comm_layer:=true`. It also
  needs the `racecar_communication` nodes running (started by the `*_racecars.launch`), or it
  subscribes to a dead `footprints_combined` topic.
- **Range gating is inconsistent** — out-of-range peers are cleared from `ids_combined`, but their
  footprint is written to `footprints_combined` unconditionally, so the costmap can mark peers the
  ID table considers out of range.
- **Hard-coded to the threeLanes world** — lane thresholds (`y ≥ 0 → 0`, `y < −0.525 → 2`), lane
  centres, and the x-range `[−50, 50]` are hard-coded in several files (`send_id_msg.py`,
  `threeLanes_current_future_pos.py`, the controllers). They silently misbehave on other maps.
- **jinja2 overwrites launch files in place** — the RL env renders `empty_world_temp.launch` and
  `one_racecar_one_ambulance_temp.launch` over the real `empty_world.launch` /
  `one_racecar_one_ambulance.launch` every episode, so those version-controlled files get clobbered
  at runtime.
- **10 episodes, ε ≈ 1.0** — with `max_num_episodes=10` and `decay_rate=1e-4`, epsilon barely
  decays, so a training run is nearly pure exploration; increase episodes / decay to actually
  exploit the learned policy.
- **Ground-truth localization** — `gazebo_odometry.py` publishes the exact Gazebo pose as
  `map → base_link`; AMCL/gmapping are provided but effectively bypassed, and their `odom_frame`
  (`/vescN/odom`) isn't wired to `base_link`, so they won't get the transform they expect without
  extra plumbing.
- **Python 2 / ROS Kinetic only** — `print` statements, `except X, e:` syntax, `np.int`/`np.float`.
  Won't run under Python 3. A stray `laneChange_action_server.cpython-312.pyc` is checked in and is
  inconsistent with the Python 2 source.
