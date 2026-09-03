# caatc — v1.0.0 line (ROS 2 / Python 3)

The ground-up rewrite of the project onto a **maintained** stack (see
[`../docs/adr/`](../docs/adr/)). The RL core is built on
[`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym) (Gymnasium API, N-agent),
pinned to the `v1.0.0` branch. Our contribution — several cars cooperating over
V2V to clear a path for an **emergency vehicle** — is layered on top (M1).

> The legacy ROS 1 / Python 2 project was removed from `master` in **M1** and is
> preserved at tags `v0.1.0`–`v0.3.0` (`git checkout v0.3.0`); its docs are in
> [`../docs/legacy/`](../docs/legacy/). See [ADR 0003](../docs/adr/0003-refactor-in-place-preserve-legacy-with-tags.md).

## Milestones (gym-first — ADR 0005)

- **M0 — scaffold & smoke** *(done)*: pin `f1tenth_gym@v1.0.0`, a Python 3 Docker
  dev image, and one headless Gymnasium episode (1 and 2 agents) to prove the base.
- **M1 — the scenario** *(done)*: `ClearanceEnv` — one **EV** + K cooperating cars,
  V2V shared observation, cooperative reward, scripted baselines, and a pre-training
  headroom gate. Legacy ROS 1 tree removed here.
- **M2 — learn it** *(done)*: train with **stable-baselines3 PPO** on the centralized
  joint action; stream episodes to the dashboard; replay any policy as a video or a
  live window. On EASY the learned policy **matches the scripted oracle** (100%
  success, 0 collisions, 6.0 s clearance, 7.44 m/s vs the naive floor's 2.29 m/s).
  See [ADR 0008](../docs/adr/0008-train-with-stable-baselines3-ppo.md).

## Run M0 (nothing installed on the host)

Needs Docker on the host — if this is a fresh machine, do the one-time
[Docker prerequisites](../README.md#prerequisites--install-docker) first (install + the
`docker` group step, which requires a fresh login).

```bash
./run.sh gym-build     # build the Python 3 image (pins f1tenth_gym @ v1.0.0)
./run.sh gym-smoke     # headless smoke: one episode, 1 and 2 agents
```

Expected: it prints the per-agent observation keys and a step count for both the
1-agent and 2-agent runs, ending with `OK: gym base runs headless.`

## M1 — ClearanceEnv

```bash
./run.sh clearance-smoke      # the pre-training headroom gate (must pass before training)
./run.sh gym-test             # unit tests (frenet / controllers / termination / headroom)
./run.sh clearance-eval --policy ideal --preset easy --episodes 20   # log a dashboard run
```

Design: [`../docs/design/m1-clearance-env.md`](../docs/design/m1-clearance-env.md) and
[ADR 0007](../docs/adr/0007-m1-clearance-env-design.md).

## M2 — train it

```bash
./run.sh train-build                                   # + stable-baselines3, CPU-only torch
./run.sh clearance-train --preset easy --timesteps 300000 --n-envs 8
./run.sh dashboard                                     # the learning curve, live

./run.sh clearance-watch --model saved_variables/models/ppo-easy.zip   # -> mp4
./run.sh view-build && ./run.sh clearance-watch --policy ideal --mode human   # a window
```

Modules: `train.py` (PPO + the dashboard callback + greedy eval through the baselines'
own path), `play.py` (replay/record), `render2d.py` (the top-down scene).
