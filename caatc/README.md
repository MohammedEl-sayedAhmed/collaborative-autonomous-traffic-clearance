# caatc — v1.0.0 line (ROS 2 / Python 3)

The ground-up rewrite of the project onto a **maintained** stack (see
[`../docs/adr/`](../docs/adr/)). The RL core is built on
[`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym) (Gymnasium API, N-agent),
pinned to the `v1.0.0` branch. Our contribution — several cars cooperating over
V2V to clear a path for an **emergency vehicle** — will be layered on top.

> The legacy ROS 1 / Python 2 project still lives in this repo (`simulator/`,
> `system/`, `run.sh sim`, …) and at tags `v0.1.0`–`v0.3.0`. It is removed in
> milestone **M1**, once this gym base is proven.

## Milestones (gym-first — ADR 0005)

- **M0 — scaffold & smoke** *(this)*: pin `f1tenth_gym@v1.0.0`, a Python 3 Docker
  dev image, and one headless Gymnasium episode (1 and 2 agents) to prove the base.
- **M1 — the scenario**: N-agent env with one **EV** + K cooperating cars, V2V
  shared observation, cooperative reward, and a scripted baseline (design with real
  headroom from day one). Legacy ROS 1 tree removed here.
- **M2 — learn it**: train with stable-baselines3 (PPO/DQN); log to the existing
  dashboard; show learned ≫ naive.

## Run M0 (nothing installed on the host)

```bash
./run.sh gym-build     # build the Python 3 image (pins f1tenth_gym @ v1.0.0)
./run.sh gym-smoke     # headless smoke: one episode, 1 and 2 agents
```

Expected: it prints the per-agent observation keys and a step count for both the
1-agent and 2-agent runs, ending with `OK: gym base runs headless.`
