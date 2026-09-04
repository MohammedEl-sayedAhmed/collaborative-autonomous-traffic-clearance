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
  - **M3 — done:** decentralized execution (ADR 0009). Each car decides from its own 26-feature view,
    one shared network evaluated K times, **no global state at execution** (`caatc/train_dec.py`,
    parameter-shared IPPO over `caatc/vec_agents.py`; optional CTDE via `--central-critic`, where the
    actor still reads only its own view). Both decision gates pass
    (`./run.sh clearance-smoke --m3`, `./run.sh dec-smoke`) — including K separate OS processes, each
    seeing only its own car's observation, reproducing the in-process metrics. **On STRICT and HARD the
    decentralized policy equals the centralized one exactly** (STRICT: 100% success, 0 collisions,
    `t_clear` 6.13 s, 3.0 yields = the oracle). On EASY plain IPPO is +9.7% on `t_clear`, and **`--central-critic` closes it to +0.33% with 3.0
    yields** — so the escalation is adopted, not merely available.
  - **The EASY gap is a scenario artifact, not a decentralization cost** (ADR 0010): the EV's ACC law
    follows whatever is ahead, so cooperators that merely SPEED UP let it through without yielding —
    100% success at ~95% of the oracle's return. **Success rate cannot distinguish cooperation from
    convoying; only clearance time can.** The **STRICT** preset caps a cooperator's speed while it is
    still in the EV's lane, which removes the substitution — and there the same learner yields 3.0/3.0
    and matches the oracle. EASY/HARD are untouched so published numbers stay valid;
    `--policy speedup` + two gate checks keep the substitution visible.
- Architectural decisions are recorded in **`docs/adr/`**. Improvement ideas in **`ROADMAP.md`**.

## Repository map
- `caatc/` — the v1.0.0 Python 3 package (RL on `f1tenth_gym`): `clearance_env.py` (M1 env),
  `scenario.py`, `frenet.py`, `controllers.py`, `baselines.py`, `clearance_eval.py`,
  `clearance_smoke.py` (headroom gate), `train.py` (M2 PPO), `play.py` + `render2d.py` (watch a
  rollout), `smoke.py` (M0), `tests/`; M3: `train_dec.py`, `decentralized.py`, `obs_spec.py`,
  `vec_agents.py`, `central_critic.py`, `pz_env.py`, `proc_fleet.py`, `dec_smoke.py`.
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
- **M3:** `./run.sh clearance-smoke --m3` (is the local view sufficient?) · `./run.sh dec-smoke`
  (plumbing + locality) · `./run.sh clearance-train-dec --preset easy|hard|strict --timesteps N`
  (`--central-critic` for CTDE) · presets are `easy|hard|strict` everywhere
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
**M4 — the ROS 2 Humble mechanical demo** (the second half of ADR 0005): bring a trained policy up on
ROS 2 Humble, mirroring the thesis's split of SUMO-for-RL and Gazebo-for-mechanics. Design it with an
ADR and confirm the open forks with the owner first, as M1 and M3 were.

Also open, in rough priority order: **richer V2V** (train against dropouts/latency rather than only
evaluating them — M3 measured graceful degradation to p=0.5 but never trained on a lossy channel);
and the ROADMAP's richer scenarios (more cars/lanes, curriculum) — K-transfer
already works at K=4.

Always run `./run.sh clearance-smoke` before training a scenario, and `--preset strict` when the claim
is about cooperation rather than mere success.
