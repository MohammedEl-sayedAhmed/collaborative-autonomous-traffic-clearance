# 0008. M2: train the ClearanceEnv with stable-baselines3 PPO (one joint policy)

- **Status:** accepted
- **Date:** 2026-09-03
- **Deciders:** Mohammed El-sayed Ahmed

## The problem

M1 delivered `ClearanceEnv` with a proven gap: a check before training shows that doing nothing is far
worse than the hand-written ideal. M2 has to *learn* that gap: produce a policy that clears the
emergency vehicle's path much better than the "nobody moves" baseline, and show it on the dashboard.

The environment offers **one joint action** for all cars, `MultiDiscrete([5] * K)`, one move-aside
choice per car, and a single shared reward. The question is which learning library and which shape of
learner to use *now*. Letting each car decide alone is a later goal (it is on the roadmap), not an M2
requirement.

## Options

- **stable-baselines3 PPO on the joint `MultiDiscrete` action.** PPO handles `MultiDiscrete`
  directly (one choice head per car), so the joint action stays `5 × K` outputs instead of `5^K`
  classes. Maintained, well documented, reproducible, runs fine on a CPU.
- **stable-baselines3 DQN.** Rejected: its DQN only takes a single `Discrete` action. The joint
  action would have to be flattened into `Discrete(5^K)`, that is 125 classes at K=3 and 625 at K=4,
  which scales badly and throws away the per-car structure. (The 2020 project's Q-table has exactly
  this shape, and it is what we are moving away from.)
- **A multi-agent method right away (MAPPO, per-car policies, PettingZoo).** Postponed: it is the
  right long-term shape, and the environment is designed so a PettingZoo mode can be added later, but
  it adds a second unproven piece on top of a brand-new environment. Learn the scenario centrally
  first, then split it per car with a working reference to compare against.
- **Our own trainer.** Rejected: the 2020 line already showed what hand-written RL code costs. Using
  a maintained library is the point of the migration.

## Decision

Use **stable-baselines3 PPO** with its default `MlpPolicy` on the joint action, in its own training
image (`docker/gym-train.Dockerfile`: the gym image plus SB3 and a **CPU-only** torch), driven by
`caatc/train.py` and `./run.sh clearance-train`.

Three things are on purpose:

1. **Training streams to the dashboard.** A callback writes every finished episode to
   `saved_variables/runs/<label>/` in the same JSONL format the M1 baselines use, so you can watch
   the learning curve live, on top of the naive / random / ideal band.
2. **Evaluation reuses the baselines' code.** The trained model is wrapped as a `policy(env) ->
   action` function and run through `clearance_eval.run_episode`, so learned and hand-written
   policies are measured by the same code. There is no second evaluation path that could drift.
3. **The headroom check comes first.** `./run.sh clearance-smoke` must pass before training. Never
   spend training time on a scenario whose gap is not proven (the 2020 lesson).

**EASY** is trained first (per ADR 0007), then **HARD**.

## Consequences

- **Good:** a standard, maintained, reproducible learner. `MultiDiscrete` keeps the joint action
  small. The dashboard comparison is fair by construction. CPU-only training keeps the Docker setup
  portable (no GPU or driver needed).
- **Cost:** the policy is **central**. One network sees all cars and outputs all their actions. That is
  not the per-car V2V story the thesis is about (that is the next milestone, and the environment is
  built for it). Training needs a heavier image (torch) and runs on the CPU, so runs are modest in
  length. PPO needs more samples than an off-policy method would.
- **Neutral:** the EV stays scripted (an M1 decision). The training image pins `gymnasium==0.29.1`
  to match the version the environment is built against.
