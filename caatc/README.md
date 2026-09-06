# caatc: the v1.0.0 line (ROS 2 / Python 3)

This is the rewrite of the project onto a **supported** stack (see [`../docs/adr/`](../docs/adr/)).
The learning code runs on [`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym) (Gymnasium API,
several cars at once), pinned to its `v1.0.0` branch. Our part, several cars cooperating over V2V to
clear a path for an **emergency vehicle**, sits on top.

> The old ROS 1 / Python 2 project was removed from `master` in **M1**. It is kept at tags `v0.1.0` to
> `v0.3.0` (`git checkout v0.3.0`), and its docs are in [`../docs/legacy/`](../docs/legacy/). See
> [ADR 0003](../docs/adr/0003-refactor-in-place-preserve-legacy-with-tags.md).

## Milestones (learning first, ROS 2 after; ADR 0005)

- **M0, the base** *(done)*: pin `f1tenth_gym@v1.0.0`, a Python Docker image, and one episode without
  a screen (1 and 2 cars) to prove the simulator runs.
- **M1, the scenario** *(done)*: `ClearanceEnv`, one **EV** plus K cooperating cars, V2V in the
  observation, a shared reward, hand-written baselines, and a headroom check before training. The old
  ROS 1 tree was removed here.
- **M2, learn it** *(done)*: train with **stable-baselines3 PPO** on the joint action, stream episodes
  to the dashboard, replay any policy as a video or in a window. On **both** EASY and HARD the learned
  policy **matches the hand-written ideal** (100% success, 0 collisions, 6.0 s clearance, about
  7.4 m/s against 2.29 m/s when nobody moves). On HARD it picks the free side from the V2V
  information, where `random` crashes 60% of the time. See
  [ADR 0008](../docs/adr/0008-train-with-stable-baselines3-ppo.md).
- **M3, each car alone** *(done)*: one shared policy, run once per car, each car seeing only its own
  26 numbers. On STRICT and HARD it gives the same numbers as the central policy; on EASY it is 1.9%
  slower with `--central-critic`. Each car can run as its own process. See
  [ADR 0009](../docs/adr/0009-decentralized-execution-ippo.md) and
  [ADR 0010](../docs/adr/0010-strict-preset-removes-the-convoying-substitution.md).
- **M4, on ROS 2** *(in progress)*: one simulator, K ROS 2 car nodes, on ROS 2 Jazzy. See
  [ADR 0011](../docs/adr/0011-m4-ros2-mechanical-demo.md).

## Run M0 (nothing installed on your machine)

You need Docker. On a fresh machine do the one-time
[Docker setup](../README.md#before-you-start-install-docker) first (install, then the `docker` group
step, which needs a new login).

```bash
./run.sh gym-build     # build the Python image (pins f1tenth_gym @ v1.0.0)
./run.sh gym-smoke     # one episode without a screen, with 1 and 2 cars
```

You should see the per-car observation keys and a step count for both runs, ending with
`OK: gym base runs headless.`

## M1: ClearanceEnv

```bash
./run.sh clearance-smoke      # the headroom check (must pass before training)
./run.sh gym-test             # unit tests
./run.sh clearance-eval --policy ideal --preset easy --episodes 20   # log a dashboard run
```

Design: [`../docs/design/m1-clearance-env.md`](../docs/design/m1-clearance-env.md) and
[ADR 0007](../docs/adr/0007-m1-clearance-env-design.md).

## M2: train it

```bash
./run.sh train-build                                   # + stable-baselines3, CPU-only torch
./run.sh clearance-train --preset easy --timesteps 300000 --n-envs 8
./run.sh dashboard                                     # the learning curve, live

./run.sh clearance-watch --model saved_variables/models/ppo-easy.zip   # -> mp4
./run.sh view-build && ./run.sh clearance-watch --policy ideal --mode human   # a window
```

Modules: `train.py` (PPO, the dashboard callback, and evaluation through the baselines' own code),
`play.py` (replay or record), `render2d.py` (the top-down drawing).

## M3: each car decides alone

```bash
./run.sh clearance-smoke --m3     # is one car's own view enough?
./run.sh dec-smoke                # does the plumbing work, and is the view really local?
./run.sh clearance-train-dec --preset strict --timesteps 900000 --n-envs 8   # --central-critic for CTDE
```

Modules: `train_dec.py`, `decentralized.py`, `obs_spec.py`, `vec_agents.py`, `central_critic.py`,
`pz_env.py`, `proc_fleet.py`, `dec_smoke.py`. Design:
[`../docs/design/m3-decentralized-execution.md`](../docs/design/m3-decentralized-execution.md).

## M4: the seam for ROS 2

`ClearanceEnv.step()` is now built from four smaller calls (`set_decision`, `joint_action_rows`,
`substep`, `commit_step`) so a ROS 2 bridge can run the physics one tick at a time and swap in the
cars' own drive commands. `tests/golden/` holds traces recorded before this change; a test replays them
and demands exact equality. Design:
[`../docs/design/m4-ros2-mechanical-demo.md`](../docs/design/m4-ros2-mechanical-demo.md).
