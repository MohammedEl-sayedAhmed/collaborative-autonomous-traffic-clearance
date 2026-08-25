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
  - **M2 — next:** train with stable-baselines3 (PPO/DQN) on `caatc/clearance-v0`; visualize on the
    dashboard; show learned ≫ naive.
- Architectural decisions are recorded in **`docs/adr/`**. Improvement ideas in **`ROADMAP.md`**.

## Repository map
- `caatc/` — the v1.0.0 Python 3 package (RL on `f1tenth_gym`): `clearance_env.py` (M1 env),
  `scenario.py`, `frenet.py`, `controllers.py`, `baselines.py`, `clearance_eval.py`,
  `clearance_smoke.py` (headroom gate), `smoke.py` (M0), `tests/`.
- `docker/` — `gym.Dockerfile` (Py3 dev image) + `gym-test.Dockerfile` (adds pytest).
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
**M2 — train it.** Add stable-baselines3 (PPO for the `MultiDiscrete` joint action) and train on
`caatc/clearance-v0` (EASY first, then HARD). Run `./run.sh clearance-smoke` first — never train a
scenario whose headroom gate does not pass. Log training to `saved_variables/runs/` (the dashboard
format `clearance_eval.write_run` already emits) and compare learned vs the naive/random/ideal band on
`./run.sh dashboard`. Then extend toward the ROADMAP items (decentralized CTDE, richer V2V).
