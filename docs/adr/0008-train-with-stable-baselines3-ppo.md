# 0008. M2 — train the ClearanceEnv with stable-baselines3 PPO (centralized joint policy)

- **Status:** accepted
- **Date:** 2026-09-03
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

M1 delivered `ClearanceEnv` with a verified performance gap (a pre-training headroom gate proves
naive ≪ ideal). M2 has to *learn* that gap: produce a policy that clears the emergency vehicle's path
markedly better than the naive baseline, and show it on the existing dashboard.

The env exposes a **centralized joint action** — `MultiDiscrete([5] * K)`, one move-aside decision per
cooperator — and a single shared cooperative reward. The question is which learning stack and which
agent decomposition to adopt *now*, given that decentralized execution (CTDE) is a stated ROADMAP goal
rather than an M2 requirement.

## Considered options

- **stable-baselines3 PPO on the joint `MultiDiscrete` action** — PPO supports `MultiDiscrete`
  natively (one categorical head per cooperator), so the joint space stays `5 × K` parameters instead
  of `5^K` classes. Maintained, well-documented, reproducible, CPU-friendly.
- **stable-baselines3 DQN** — rejected: SB3's DQN only supports `Discrete`. It would need the joint
  action flattened to `Discrete(5^K)` = 125 classes at K=3 (and 625 at K=4), which scales badly and
  discards the per-agent structure. (The 2020 project's tabular Q-learning is the same shape and is
  exactly what we are moving beyond.)
- **A multi-agent CTDE algorithm now (MAPPO / per-agent policies + PettingZoo)** — deferred: it is the
  right long-term shape (and the env is designed to expose a PettingZoo-parallel mode later), but it
  adds a second unproven axis on top of a freshly built env. Learn the scenario centrally first, then
  decentralize with a working reference to compare against.
- **A hand-rolled trainer** — rejected: the 2020 line already showed the cost of bespoke RL code; a
  maintained implementation is the point of the migration.

## Decision

Adopt **stable-baselines3 PPO** with the default `MlpPolicy` on the centralized joint action, in a
dedicated training image (`docker/gym-train.Dockerfile`: the gym image + SB3 + a **CPU-only** torch
wheel), driven by `caatc/train.py` and `./run.sh clearance-train`.

Three properties are deliberate:

1. **Training streams to the dashboard.** A callback writes each finished episode to
   `saved_variables/runs/<label>/` in the same JSONL format the M1 baselines emit, so the learning
   curve is watchable live and overlays directly on the naive / random / ideal band.
2. **Evaluation reuses the baselines' own path.** The trained model is wrapped as a
   `policy(env) -> action` callable and run through `clearance_eval.run_episode`, so learned and
   scripted policies are measured by identical code — no separate eval path to drift.
3. **The headroom gate is a precondition.** `./run.sh clearance-smoke` must pass before training;
   never spend learning on a scenario whose gap is unproven (the lesson of the legacy toy harness).

**EASY** is trained first (per ADR 0007), then **HARD**.

## Consequences

- **Positive:** a standard, maintained, reproducible learner; `MultiDiscrete` keeps the joint action
  compact; the dashboard comparison is apples-to-apples by construction; CPU-only training keeps the
  containerized workflow portable (no GPU/driver dependency).
- **Negative / trade-offs:** the policy is **centralized** — one network sees all cooperators and emits
  all their actions, which is not the decentralized V2V story the thesis ultimately targets (that is
  the CTDE ROADMAP item, and the env already anticipates it). Training adds a heavier image (torch)
  and is CPU-bound, so run lengths are modest. PPO's on-policy sample cost is higher than an off-policy
  learner's would be.
- **Neutral:** the EV stays scripted (an M1 decision); SB3 pins `gymnasium==0.29.1` in the training
  image to match the version the env is built against.
