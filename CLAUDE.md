# CLAUDE.md — working guide for this repository

## What this is
**Collaborative Autonomous Traffic Clearance** — cooperative multi-agent reinforcement learning where
several 1/10-scale autonomous cars clear a path for an **emergency vehicle** over V2V communication.
Originally a 2020 graduation project (ROS Kinetic / Gazebo 7 / Python 2, on the F1TENTH / MIT racecar
stack); now being migrated to a maintained **ROS 2 Humble / Python 3** stack built on `f1tenth_gym`.

## Current status — read this first
- **Legacy ROS 1 / Python 2 line** is frozen at tags **`v0.1.0`** (baseline), **`v0.2.0`** (fixed &
  reproducible), **`v0.3.0`** (enhanced). `git checkout v0.3.0` to see it. It still lives on `master`
  (`simulator/`, `system/`, `docker/`, `run.sh sim|nav|movecar|ev|campaign|rl-blocker`, `tools/`) and
  is **removed in milestone M1**.
- **v1.0.0 migration (ROS 2 Humble / Python 3)** is in progress, **gym-first** (see `docs/adr/`):
  - **M0 — done:** the `caatc/` package on **f1tenth_gym v1.0.0** (Gymnasium API, N-agent), Dockerized
    and headless. Prove it: `./run.sh gym-build && ./run.sh gym-smoke`.
  - **M1 — next:** a `ClearanceEnv` wrapper — one agent is the emergency vehicle, the others cooperate
    to clear its path (V2V observation + cooperative reward), on a scenario with **guaranteed
    headroom**. Full design in **`docs/design/m1-clearance-env.md`** + **ADR 0007**. M1 also removes
    the legacy ROS 1 tree (ADR 0003). *There is one open design fork to confirm — see the design doc.*
  - **M2:** train with stable-baselines3 (PPO/DQN); visualize on the existing dashboard.
- Architectural decisions are recorded in **`docs/adr/`**. Improvement ideas in **`ROADMAP.md`**.

## Repository map
- `caatc/` — the v1.0.0 Python 3 package (new work; RL on `f1tenth_gym`).
- `docs/adr/` — Architecture Decision Records (the migration decisions and rationale).
- `docs/design/` — design docs (e.g. the M1 ClearanceEnv design).
- `docs/` — legacy architecture / running / dashboard / RL-experiment guides.
- `tools/` — `rl_harness` (legacy headless RL harness), `dashboard` (a stack-agnostic training
  dashboard that reads JSONL runs — carries over to the v1.0.0 line unchanged).
- `simulator/`, `system/`, `docker/`, `docker-compose.yml`, `run.sh` (ROS parts) — **legacy ROS 1**
  stack (removed in M1).
- `thesis/` — the graduation thesis, a **private git submodule**. Public clones get only the pointer;
  `git submodule update --init thesis` needs access to the private thesis repo. Build: `./run.sh thesis`.

## Key commands — everything runs in Docker; nothing is installed on the host
- **v1.0.0 (gym):** `./run.sh gym-build` · `./run.sh gym-smoke`
- **Legacy sim / RL:** `./run.sh sim` · `./run.sh dashboard` · `./run.sh campaign` · `./run.sh rl-blocker`
- **Thesis:** `./run.sh thesis`
- `./run.sh` with no arguments prints every command.

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
Confirm the single open design fork in `docs/design/m1-clearance-env.md` (blocking mechanism +
first-preset), then implement M1 (the `ClearanceEnv` wrapper, baselines, and the pre-training
headroom gate) and remove the legacy ROS 1 tree in its own commit.
