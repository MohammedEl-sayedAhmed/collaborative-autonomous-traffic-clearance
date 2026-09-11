# CLAUDE.md: working guide for this repository

## What this is
**Collaborative Autonomous Traffic Clearance**: several 1/10-scale self-driving cars learn to clear a
path for an **emergency vehicle**, talking to each other over V2V (vehicle-to-vehicle radio). It was a
2020 graduation project on ROS Kinetic / Gazebo 7 / Python 2 (the F1TENTH / MIT racecar stack). It is
now rebuilt on a supported **ROS 2 Jazzy / Python 3.12** stack on top of `f1tenth_gym`.

## Where things stand (read this first)
- **The old ROS 1 / Python 2 line** is frozen at tags **`v0.1.0`** (as it was), **`v0.2.0`** (fixed and
  reproducible), **`v0.3.0`** (improved). `git checkout v0.3.0` to see it. It was **removed from
  `master` in M1** (ADR 0003). Its docs are in **`docs/legacy/`**.
- **The v1.0.0 rewrite (ROS 2 Jazzy / Python 3.12)** is complete, M0 to M4, built learning first (see `docs/adr/`):
  - **M0, done:** the `caatc/` package on **f1tenth_gym v1.0.0** (Gymnasium API, several cars), in
    Docker, no screen needed. Check: `./run.sh gym-build && ./run.sh gym-smoke`.
  - **M1, done:** the `ClearanceEnv` wrapper. `agent_0` is the scripted emergency vehicle; K
    cooperators clear its path (V2V in the observation, one shared reward). The headroom is
    **guaranteed** by an adaptive-cruise (ACC) law on wide lanes. Baselines (naive / random / ideal), a
    JSONL dashboard evaluator, a **headroom check before training** (`./run.sh clearance-smoke`, fails
    if the gap is missing), and unit tests (`./run.sh gym-test`). Design:
    **`docs/design/m1-clearance-env.md`** and **ADR 0007**. M1 also removed the old ROS 1 tree (ADR 0003).
  - **M2, done:** trained with **stable-baselines3 PPO** on one joint `MultiDiscrete` action
    (`caatc/train.py`, ADR 0008), streaming every episode to the dashboard and scoring the trained
    policy through the *baselines' own* evaluation code. On **both** presets the learned policy
    **matches the hand-written ideal**: EASY 100% success / 0 collisions / `t_clear` 6.00 s / EV
    7.44 m/s / return 102.3 (naive: 0% / 2.29 m/s / 31.4); HARD 100% / 0% / 6.00 s / 7.43 m/s / 102.3,
    where `random` crashes 60% of the time. It learned to read the V2V information and merge to the
    *free* side. Exactly K=3 lane changes, no wobbling. Also a **top-down scene drawing**
    (`caatc/render2d.py`) and a replay tool (`caatc/play.py`): record an mp4 without a screen, or open
    a live window.
  - **M3, done:** each car decides alone (ADR 0009). Each car acts on its own 26 numbers, one shared
    network run K times, **no shared state at run time** (`caatc/train_dec.py`, shared IPPO over
    `caatc/vec_agents.py`; optional `--central-critic`, where the critic sees everything during training
    but each car's actor still reads only its own view). Both checks pass (`./run.sh clearance-smoke
    --m3`, `./run.sh dec-smoke`), including K separate operating-system processes, each seeing only its
    own car's observation, reproducing the in-process numbers. **On STRICT and HARD the per-car policy
    equals the central one exactly** (STRICT: 100% success, 0 collisions, `t_clear` 6.13 s, 3.0 yields,
    the same as the ideal). On EASY plain IPPO is 9.7% slower on `t_clear`, and **`--central-critic`
    brings that to +1.9%**, so that option is the recommended one on EASY.
  - **The EASY gap comes from the scenario, not from splitting per car** (ADR 0010): the EV's ACC law
    follows whatever is ahead, so cars that simply SPEED UP let it through without moving aside: 100%
    success at about 95% of the ideal's return. **The success rate cannot tell cooperation from a fast
    convoy; only the clearance time can.** The **STRICT** preset caps a car's speed while it is still in
    the EV's lane, which closes that loophole, and there the same learner yields 3.0 out of 3.0 and
    matches the ideal. EASY and HARD are unchanged so the published numbers stay valid; `--policy
    speedup` plus two checks keep the loophole visible.
  - **M4, done:** the ROS 2 Jazzy demo (ADR 0011, ADR 0012). Python 3.12
    everywhere with all 17 published rows re-checked (one model retrained and re-measured); the
    **seam** in `ClearanceEnv` (`set_decision` / `joint_action_rows` / `substep` / `commit_step`) with
    golden traces and tests proving `step()` did not change by a single bit; the `caatc-ros` image on
    Jazzy, a numerical twin (`./run.sh ros-fingerprint --gate` says IDENTICAL); and **M4.1: one car
    drives over ROS 2 in lockstep** (`./run.sh ros-smoke`): the ROS run and the headless run agree on
    every tick, the replay is exact, and the node's 26 numbers equal the bridge's exactly. The brains
    are pure Python (`caatc/ros_node_core.py`, `caatc/ros_bridge_core.py`), tested without ROS; the
    rclpy shells live in `ros2/src/caatc_ros/`. **M4.2 done too:** all K cars over the V2V relay on
    the exported numpy policy (`caatc/policies/ippo-strict`), a gate node for the allow-list, and
    `./run.sh ros-gate` running every check (fleet on strict/hard/easy, naive and speedup must fail on
    strict, the stress run). **M4.3 done:** rosbag replay (check 8), the RViz scene (`ros-demo` +
    `ros-view 42`), `ros-video`, and a dashboard run. **M4.4 done:** the bridge's `--async` mode
    (nobody waits; check 13 measures command age, held ticks and the real-time factor: at real time
    about 8% of ticks reuse a 10 ms old command and the outcome is unchanged, `./run.sh ros-async`) and
    the loss/delay sweep (`./run.sh ros-sweep`): on STRICT the policy yields even with a blind radio;
    on HARD a lost or 500 ms old broadcast makes an occupied lane look empty and the cars collide. The
    tables and the M4.1 / M4.2 contracts are in the design doc; ADR 0011 has the outcome.
- Big decisions and their reasons are in **`docs/adr/`**. Ideas for later are in **`ROADMAP.md`**.

## Repository map
- `caatc/`: the v1.0.0 Python package (learning on `f1tenth_gym`): `clearance_env.py` (the M1 scenario
  + the M4 seam), `scenario.py`, `frenet.py`, `controllers.py`, `baselines.py`, `clearance_eval.py`,
  `clearance_smoke.py` (the headroom check), `train.py` (M2 PPO), `play.py` + `render2d.py` (watch a
  run), `smoke.py` (M0), `tests/` (with `golden/` traces recorded before the seam); M3: `train_dec.py`,
  `decentralized.py`, `obs_spec.py`, `vec_agents.py`, `central_critic.py`, `pz_env.py`,
  `proc_fleet.py`, `dec_smoke.py`; M4: `actions.py`, `ros_tick.py`, `ros_geometry.py`,
  `ros_fingerprint.py` (check 1), `ros_node_core.py`, `ros_bridge_core.py`, `ros_v2v.py` (the relay's
  brain), `policy_export.py` + `policies/` (the exported actors).
- `ros2/src/`: `caatc_msgs` (Episode, Decision, Broadcast, V2VDigest) and `caatc_ros` (the bridge, the
  car node, the relay, the gate, `bag_replay`, `scene_view`, `msgs_io`, the `ros_smoke` orchestrator,
  `rviz/clearance.rviz`). `docker/ros.Dockerfile` builds `caatc-ros`; `docker/ros-view.Dockerfile`
  adds RViz for viewing. `caatc/ros_video.py` draws a record as an mp4.
- `docker/`: `gym.Dockerfile` (Python dev image), `gym-test.Dockerfile` (pytest),
  `gym-train.Dockerfile` (stable-baselines3 + CPU torch), `gym-view.Dockerfile` (X11 for a window).
- `run.sh`: runs everything in Docker (v1.0.0 only: `gym-*`, `clearance-*`, `dashboard`, `thesis`).
- `docs/adr/`: Architecture Decision Records (each migration decision and why).
- `docs/design/`: design docs (M1, M3, M4, and the M5 plan).
- `gazebo/`: M5's Gazebo assets: `models/racecar` (SDF), `worlds/`, `spike.py`. `docker/gazebo.Dockerfile` builds
  `caatc-gazebo` (caatc-ros + Gazebo Harmonic, headless). `caatc/plant.py` is the plant interface (`GymPlant`).
- `docs/legacy/`: docs for the tagged 2020 ROS 1 / Gazebo stack.
- `docs/DASHBOARD*.md`: the training dashboard guides.
- `docs/upstream-candidates.md`: four `f1tenth_gym` findings. **Nothing has been sent upstream.**
- `tools/dashboard/`: the training dashboard, reads JSONL runs from `saved_variables/runs/`.
- `thesis/`: the graduation thesis, a **private git submodule**. Public clones get only the pointer;
  `git submodule update --init thesis` needs access to the private repo. Build: `./run.sh thesis`.
- The **old ROS 1 stack** (`simulator/`, `system/`, ROS `docker/`, `docker-compose.yml`,
  `tools/rl_harness/`) was removed from `master` in M1. It lives at tags `v0.1.0` to `v0.3.0`.

## Key commands (everything runs in Docker; nothing is installed on the machine)
- **v1.0.0 base:** `./run.sh gym-build` · `./run.sh gym-smoke`
- **M1:** `./run.sh clearance-smoke` (headroom check) · `./run.sh gym-test` (unit tests) ·
  `./run.sh clearance-eval --policy naive|random|speedup|ideal --preset easy|hard|strict`
- **M2:** `./run.sh train-build` · `./run.sh clearance-train --preset easy --timesteps 300000 --n-envs 8`
  · `./run.sh clearance-watch [--model <zip>|--policy ideal] [--mode human]` (a window needs
  `./run.sh view-build`). Never train unless `clearance-smoke` passes.
- **M3:** `./run.sh clearance-smoke --m3` (is one car's own view enough?) · `./run.sh dec-smoke`
  (plumbing + locality) · `./run.sh clearance-train-dec --preset easy|hard|strict --timesteps N`
  (`--central-critic` for the central critic). Presets are `easy|hard|strict` everywhere.
- **M4 (ROS 2):** `./run.sh ros-build` · `./run.sh ros-fingerprint --gate` (check 1) ·
  `./run.sh ros-smoke --preset easy|hard|strict --seeds 0,1 [--stress]` (one car) ·
  `./run.sh ros-fleet` (all K cars, relay, learned policy, gate, bag, dashboard) · `./run.sh ros-gate`
  (every check) · `./run.sh ros-video <record.npz>` · `./run.sh ros-view-build` + `./run.sh ros-demo`
  and `./run.sh ros-view 42` (RViz) · `./run.sh ros-async` (check 13, nobody waits) ·
  `./run.sh ros-sweep [--preset hard --seeds 0,1,2,3,4 --losses 0,0.5,1 --delays 0,10,50]` (loss and
  delay tables) · `./run.sh export-policy <zip> --out caatc/policies/<name>` · `./run.sh ros-shell`. Nodes always start as `python3 -m caatc_ros.<node>` inside the image, never
  via `ros2 run`.
- **Dashboard:** `./run.sh dashboard` (reads `saved_variables/runs/`) · `./run.sh dashboard-demo`
- **Thesis:** `./run.sh thesis`
- `./run.sh` with no arguments lists every command.
- **Old commands** (`sim`, `nav`, `campaign`, …) are at tag `v0.3.0` (`git checkout v0.3.0`).

## How we work
- **Docker only.** Never install project dependencies on the machine. Dependencies live in Docker,
  build results in volumes.
- **Small, single-purpose commits.** Work on a branch, open a PR, merge (this is the owner's personal
  GitHub). Conventional-commit-style messages. **No AI attribution in commit messages.**
- **Plain English everywhere** (README, docs, ADRs, commit messages, PR text): short sentences,
  everyday words, explain a term the first time it appears. Write as the person doing the work.
- **Rebuild in place.** No new repo, no `v2/` folder. The old stack is kept at the `v0.x` tags (ADR 0003).
- Before changing old behaviour, check the **thesis** for the history (why a choice was made), but treat
  the thesis as a reference that can be wrong, not as the final word.
- The network here is flaky. Retry git pushes and pulls and Docker image builds.

## Next step
**M5: a 3D plant behind the same seam, then one real car** (ADR 0013, plan in
`docs/design/m5-3d-plant.md`; decided 2026-09-11, not started). `master` is tagged `v1.0.0` (M0 to M4).
The four choices are made: an F1TENTH-style 1/10 car (one of them), lidar plus a map for positioning
(AMCL), **Gazebo Harmonic** (the official Jazzy pairing, supported to May 2029 like Jazzy; Lyrical plus
Jetty, to 2031, is the recorded upgrade path), and **nothing retrained** in M5. Work in order: (0) ~~the
M5.0 spike~~ done: `caatc-gazebo` image (`./run.sh gazebo-build`), `gazebo/` (car model, worlds,
`spike.py`; `./run.sh gazebo-spike --world spike4`): above real time with four lidars, bit-identical
repeats; the `Plant` split (`caatc/plant.py`, `GymPlant`) bit-identical on the golden traces. Gotchas:
keep Gazebo subscriptions and blocking requests in separate Python processes, sync on the world clock
topic, no `<topic>` on per-model plugins, a readiness probe must say `pause: true`; (1) M5.1: `GazeboPlant` with ground-truth poses, the generated
world, the M4 checks and tables re-run; (2) M5.2: simulated lidar, IMU and wheel odometry, AMCL per
car, the localization error published; (3) M5.3, **only if a real car is obtained** (there is none as of 2026-09-11): measure the car, map the
track, a `lab` preset, one real cooperator in the episode; (4) the record. **No end-of-life dependency anywhere**, and nothing
that only works on Gazebo Harmonic.

Still open after M5: a realistic radio (roadmap 20; M4.4 showed on HARD that "not heard" must not mean
"not there"), the bigger scenarios (10), CI (16).

M4's idea in one line, for context: keep **one** `ClearanceEnv` as the only physics, and let **K ROS 2
nodes, one per car**, replace exactly rows `1..K` of the per-tick action array. Nothing about the EV,
the ACC law, the reward or the metrics changes, so the published numbers stay comparable. M5 keeps that
and only puts a second physics behind the seam.

The three choices: **ROS 2 Jazzy** (supported to May 2029; Humble stops in May 2027, ADR 0012), which
means **Python 3.12 everywhere**; **standard ROS messages only** (`nav_msgs/Odometry`,
`ackermann_msgs/AckermannDriveStamped`), so faithfulness is checked against a *declared tolerance*
rather than bit for bit; the policy **exported to numpy** so the robot image needs no torch; and **no
end-of-life dependencies at all** (nothing copied from `f1tenth_gym_ros`).

Work in order: (1) ~~Python 3.12 and re-check every published table~~ done; (2) ~~the seam in
`ClearanceEnv` with tests proving nothing changed~~ done; (3) ~~the `caatc-ros` image on `ros:jazzy`,
with the headroom check run inside it~~ done (IDENTICAL); (4) ~~one car node in lockstep~~ done
(0 differing ticks); (5) ~~the full fleet, the V2V relay, the exported numpy policy and the remaining
checks~~ done (`./run.sh ros-gate` all green); (6) ~~the RViz view, a rosbag and a dashboard run~~ done;
(7) ~~the async mode with measured delay and loss~~ done (`ros-async`, `ros-sweep`; the tables are in
the design doc); (8) ~~the final docs pass~~ done; then the PR to master.

Also open: a richer V2V model (train against drops and delay, not just measure them; M4.4 showed why:
on HARD a lost broadcast reads as an empty lane) and the bigger scenarios in the roadmap. `docs/upstream-candidates.md` holds four `f1tenth_gym` findings; **nothing has
been sent upstream**, that is a separate decision.

Always run `./run.sh clearance-smoke` before training, and use `--preset strict` when the claim is about
cooperation rather than mere success.
