# CLAUDE.md — working guide for this repository

## What this is
**Collaborative Autonomous Traffic Clearance** — cooperative multi-agent reinforcement learning where
several 1/10-scale autonomous cars clear a path for an **emergency vehicle** over V2V communication.
Originally a 2020 graduation project (ROS Kinetic / Gazebo 7 / Python 2, on the F1TENTH / MIT racecar
stack); now being migrated to a maintained **ROS 2 Humble / Python 3** stack built on `f1tenth_gym`.

## Current status — read this first
- **Legacy ROS 1 / Python 2 line** is frozen at tags **`v0.1.0`** (baseline), **`v0.2.0`** (fixed &
  reproducible), **`v0.3.0`** (enhanced). `git checkout v0.3.0` to see it. It was **removed from
  `master` in M1** (ADR 0003); its docs are in **`docs/legacy/`**.
- **v1.0.0 migration (ROS 2 Humble / Python 3)** is in progress, **gym-first** (see `docs/adr/`):
  - **M0 — done:** the `caatc/` package on **f1tenth_gym v1.0.0** (Gymnasium API, N-agent), Dockerized
    and headless. Prove it: `./run.sh gym-build && ./run.sh gym-smoke`.
  - **M1 — done:** the `ClearanceEnv` wrapper — `agent_0` is the scripted emergency vehicle, K
    cooperators clear its path (V2V observation + cooperative reward), with **guaranteed headroom** via
    an **ACC-on-wide-lanes** blocking law (both design forks confirmed: ACC + EASY-first). Baselines
    (naive/random/ideal), a JSONL dashboard evaluator, a **pre-training headroom gate**
    (`./run.sh clearance-smoke`, exits non-zero if the gap is absent), and unit tests
    (`./run.sh gym-test`). Design: **`docs/design/m1-clearance-env.md`** + **ADR 0007**. M1 also
    removed the legacy ROS 1 tree (ADR 0003).
  - **M2 — done:** trained with **stable-baselines3 PPO** on the centralized `MultiDiscrete` joint
    action (`caatc/train.py`, ADR 0008), streaming every episode to the dashboard and evaluating the
    greedy policy through the *baselines' own* eval path. On **both** presets the learned policy
    **matches the scripted oracle**: EASY 100% success / 0 collisions / `t_clear` 6.00 s / EV
    7.44 m/s / return 102.3 (naive 0% / 2.29 m/s / 31.4); HARD 100% / 0% / 6.00 s / 7.43 m/s /
    102.3 where `random` collides 60% of the time — i.e. it learned to read the V2V occupancy and
    merge to the *free* side. Exactly K=3 lane changes, no oscillation. Also added a **top-down scene renderer**
    (`caatc/render2d.py`) + replay tool (`caatc/play.py`): record an mp4 headless, or a live window.
- Architectural decisions are recorded in **`docs/adr/`**. Improvement ideas in **`ROADMAP.md`**.

## Repository map
- `caatc/` — the v1.0.0 Python 3 package (RL on `f1tenth_gym`): `clearance_env.py` (M1 env),
  `scenario.py`, `frenet.py`, `controllers.py`, `baselines.py`, `clearance_eval.py`,
  `clearance_smoke.py` (headroom gate), `train.py` (M2 PPO), `play.py` + `render2d.py` (watch a
  rollout), `smoke.py` (M0), `tests/`.
- `docker/` — `gym.Dockerfile` (Py3 dev image), `gym-test.Dockerfile` (pytest),
  `gym-train.Dockerfile` (stable-baselines3 + CPU torch), `gym-view.Dockerfile` (X11 for a window).
- `run.sh` — the containerized runner (v1.0.0 only: `gym-*`, `clearance-*`, `dashboard`, `thesis`).
- `docs/adr/` — Architecture Decision Records (the migration decisions and rationale).
- `docs/design/` — design docs (the M1 ClearanceEnv design).
- `docs/legacy/` — docs for the tagged 2020 ROS 1 / Gazebo stack (architecture / subsystems /
  packages / running / known-issues / RL-experiments).
- `docs/DASHBOARD*.md` — the training dashboard guides.
- `tools/dashboard/` — a stack-agnostic training dashboard that reads JSONL runs (carries over
  unchanged; reads `saved_variables/runs/`).
- `thesis/` — the graduation thesis, a **private git submodule**. Public clones get only the pointer;
  `git submodule update --init thesis` needs access to the private thesis repo. Build: `./run.sh thesis`.
- **Legacy ROS 1 stack** (`simulator/`, `system/`, ROS `docker/`, `docker-compose.yml`,
  `tools/rl_harness/`) was removed from `master` in M1 — it lives at tags `v0.1.0`–`v0.3.0`.

## Key commands — everything runs in Docker; nothing is installed on the host
- **v1.0.0 (gym):** `./run.sh gym-build` · `./run.sh gym-smoke`
- **M1:** `./run.sh clearance-smoke` (headroom gate) · `./run.sh gym-test` (unit tests) ·
  `./run.sh clearance-eval --policy naive|random|ideal --preset easy|hard`
- **M2:** `./run.sh train-build` · `./run.sh clearance-train --preset easy --timesteps 300000 --n-envs 8`
  · `./run.sh clearance-watch [--model <zip>|--policy ideal] [--mode human]` (needs `./run.sh view-build`
  for a window) — never train unless `clearance-smoke` passes.
- **Dashboard:** `./run.sh dashboard` (reads `saved_variables/runs/`) · `./run.sh dashboard-demo`
- **Thesis:** `./run.sh thesis`
- `./run.sh` with no arguments prints every command.
- **Legacy commands** (`sim`, `nav`, `campaign`, …) live at tag `v0.3.0` (`git checkout v0.3.0`).

## Working conventions
- **Containerized only** — never install project dependencies on the host; deps live in Docker,
  build artifacts in volumes.
- **Atomic commits**; work on a branch → open a PR → merge (this is the owner's personal GitHub).
  Conventional-commit-style messages; **no AI attribution** in commit messages.
- **Migrate in place** — no new repo, no `v2/` directory; the legacy stack is preserved via the
  `v0.x` tags (ADR 0003).
- Before changing legacy behavior, check the **thesis** for the back-history (why a choice was made) —
  but treat the thesis as a fallible reference, not an authority.
- The network here is flaky — retry git pushes/pulls and Docker image builds.

## Next step
**M3 — decentralize.** Both presets are solved centrally (M2). The open work, in rough order: move
from one centralized joint policy to **CTDE / per-agent policies** (the env already
anticipates a PettingZoo-parallel obs mode) so execution is decentralized like the thesis intends;
then richer V2V (intention sharing, dropouts) and the **ROS 2 Humble mechanical demo** (ADR 0005).
Always run `./run.sh clearance-smoke` before training a scenario.
