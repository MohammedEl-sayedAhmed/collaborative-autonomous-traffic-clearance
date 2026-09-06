# 0009. M3: each car decides alone, with one shared policy (IPPO)

- **Status:** accepted
- **Date:** 2026-09-03 (open questions decided 2026-09-03)
- **Deciders:** Mohammed El-sayed Ahmed

## The problem

M2 gave us one central PPO policy over the joint action that matches the hand-written ideal on both
presets (100% success, 0 collisions, `t_clear` 6.00 s, return 102.3). But that controller is **one
brain for all cars** at run time. That is not the 2020 thesis's story, where each car runs its own
model and coordinates over V2V, and it is not the headline roadmap item. [ADR
0008](0008-train-with-stable-baselines3-ppo.md) put off the per-car split on purpose, so we could try
it against a working reference.

That reference now exists, and it already sits at the ideal. So M3 **cannot win on the main
numbers**. Its job is to make "each car decides alone" *provable*, and to add what one central brain
cannot do: run with a different number of cars, and cope with lost V2V messages per car.

What makes this feasible: the central observation is already the K per-car observations glued
together, and each per-car observation has the same 26 numbers whatever K is. Splitting it per car is
a reshape, not a rewrite.

## Options

1. **Parameter-shared IPPO on stable-baselines3.** IPPO means independent PPO learners; "parameter
   shared" means they all use one network. We split the K per-car streams out of one joint scenario
   with a `VecEnvWrapper`. One shared actor and critic, five actions per car, deployed as K separate
   forward passes. **Chosen.** No new dependencies. Training and deployment see the same thing. All K
   experiences per physics step are used.
2. **Our own MAPPO with a critic that sees everything.** Rejected: ADR 0008 rejected hand-written RL
   code, and we expected a central critic to help little here. Kept as a *recorded next step* (each
   car's actor reads only its own view; the critic reads all K views) in case IPPO stalls.
3. **Take turns training one car against frozen partners** (a "partner ladder" plus a cross-play
   table). Rejected as the learner: training and deployment differ until the last stage, it can go in
   circles with no fix in scope, and it throws away K−1 of every K experiences. We did take over its
   *interfaces*, its local-oracle check and its `LocalOnlyView` guard.
4. **PettingZoo + SuperSuit.** Rejected for training: training does not go through a `ParallelEnv`
   here, so it would only add pins and unused code. `per_agent_obs_all()` is the hook an adapter can
   wrap later.
5. **A realistic lossy V2V radio first** (drops, delay, sharing intentions). Postponed to M4: it
   changes the observation size, so the M2 reference stops being comparable, and it would measure a
   train/test mismatch rather than the per-car split. *Evaluating* under message loss is still in M3.

## Decision

Use **parameter-shared IPPO** with per-car observations and a `SharedPolicySquad` wrapper for
deployment, behind two checks that can fail, as laid out in
[`../design/m3-decentralized-execution.md`](../design/m3-decentralized-execution.md):

- **Step 0 proves the idea before any training:** a hand-written *local* policy that sees only the 26
  per-car numbers must match the all-seeing ideal, through the same `run_episode` code. If it cannot,
  M3 is the wrong milestone and the right one is a richer observation.
- **A locality check that can fail:** moving a car that is out of radio range must not change any
  per-car observation. **This failed at first**: only the EV part of the observation was range-limited,
  while the two nearest-neighbour slots were filled from anywhere on the road, so a car 40 m away still
  showed up in a "local" slot (roadmap item 19, found again in the new stack, inside the very feature
  the per-car story rests on).
- **Success means: the same result with less information**, measured on shared seeds against M2, plus
  the two new columns (a different K, lost messages). It is not an "it does better" claim.

**All three open questions were decided on 2026-09-03:**

1. **The learner:** parameter-shared IPPO over the per-car split (not the partner ladder, not MAPPO
   now; the central-critic version stays the recorded next step).
2. **Range-limit the neighbour slots now** (`neighbor_range = None → v2v_range`), so "no car sees the
   whole picture" is true by construction and the locality check can actually fail. Measured to change
   nothing on any setting the M2 comparison uses, and a test proves the numbers are identical.
3. **Build the PettingZoo mode in M3**, rather than later, to keep ADR 0007's promise. It is a
   *tested* adapter over `per_agent_obs_all()`, the standard interface for outside multi-agent
   libraries (QMIX, MADDPG; roadmap item 6), with `pettingzoo` as an optional, test-only dependency.
   Training still goes through `AgentSplitVecEnv`, so the adapter has its own conformance tests
   rather than sitting unused.

## Consequences

- **Good:** each car decides alone, as the thesis intends, and the property is *checked*, not just
  claimed. No new dependencies. The M2 path stays as the reference. The per-car action becomes
  `Discrete(5)`, so off-policy learners become possible later. Running with a different K, and
  robustness to lost messages, become measurable for the first time.
- **Cost:** independent learners have no central critic, so credit for a good outcome rests on the
  shared reward (the recorded next step addresses this). One shared policy means all cars behave the
  same; there are no specialised cars. The `AgentSplitVecEnv` plumbing (about 110 lines) is ours to
  maintain, and it must keep the per-episode dashboard numbers on the team scale.
- **Neutral:** the EV stays scripted. V2V is lossless during training (message loss is evaluated, not
  trained against, until M4).
