# 0009. M3 — decentralized execution via a parameter-shared per-agent policy (IPPO)

- **Status:** proposed
- **Date:** 2026-09-03
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

M2 delivered one centralized PPO over the joint `MultiDiscrete([5]*K)` action that matches the scripted
oracle on both presets (100% success, 0 collisions, `t_clear` 6.00 s, return 102.3). The controller is
therefore **centralized at execution time**, which is neither the 2020 thesis's story — each car runs
its own trained model and coordinates over V2V — nor the headline ROADMAP item. [ADR
0008](0008-train-with-stable-baselines3-ppo.md) deliberately deferred decentralization so it could be
attempted against a working reference.

That reference now exists, and it sits at the oracle ceiling. So M3 **cannot win on the primary
metrics**; its job is to make the decentralization property itself *verifiable*, and to deliver the
capabilities a joint controller structurally cannot (transfer across K, per-agent communication loss).

The enabling fact: `_build_obs` is already `concat(_per_coop_obs(0..K-1))`, and the per-agent width
`_F = 26` is independent of K. Decentralization is a reshape, not a rewrite.

## Considered options

1. **Parameter-shared IPPO on stable-baselines3**, with K single-agent streams split out of one joint
   scenario env by a `VecEnvWrapper` — **chosen**. One shared actor/critic, `Discrete(5)` per agent,
   deployed as K independent forward passes. Zero new dependencies; the training condition equals the
   deployment condition; all K transitions per physics step are harvested.
2. **A hand-rolled MAPPO with a privileged critic** — rejected: ADR 0008 rejected bespoke RL code, and
   at this coupling a centralized critic is predicted to buy little. Kept as a *recorded escalation*
   (actor reads only its own features, critic reads the joint vector) if IPPO plateaus.
3. **Frozen-partner iterated best response** (a partner ladder + cross-play matrix) — rejected as the
   learner: deployment ≠ training condition until the last stage, cycling has no in-scope remedy, and
   it discards K−1 of every K transitions. Its *interfaces*, its local-oracle gate and its
   `LocalOnlyView` enforcement are adopted wholesale.
4. **PettingZoo + SuperSuit** — rejected: training does not route through a `ParallelEnv` in this
   design, so it would add pins and dead code; `per_agent_obs_all()` is the seam an adapter wraps later.
5. **A realistic lossy V2V channel first** (dropouts, latency, intention sharing) — deferred to M4: it
   changes the observation width, so the M2 reference stops being comparable, and it would measure a
   train/test shift rather than decentralization. Dropout *evaluation* is still in M3 scope.

## Decision

Adopt **parameter-shared IPPO** with per-agent observations and a `SharedPolicySquad` deployment
wrapper, staged behind two decision gates, per
[`../design/m3-decentralized-execution.md`](../design/m3-decentralized-execution.md):

- **Stage 0 proves the premise before any training:** a hand-written *local* oracle, seeing only the
  26-feature per-agent view, must match the privileged oracle through the same `run_episode` path. If it
  cannot, M3 is the wrong milestone and the right one is observation enrichment.
- **A locality gate that can fail:** displacing an out-of-range car must not change any per-agent
  observation. This **fails today** — only the EV block is range-gated while the M=2 neighbour block
  fills unconditionally, so a car 40 m away still lands in a "local" slot (ROADMAP #19 reproduced in
  the new stack, inside the feature the decentralization story rests on).
- **Success is equality under information restriction**, measured on shared eval seeds against M2, plus
  the two capability columns (K-transfer, dropout). Not an improvement claim.

Status is **proposed** pending the owner's confirmation of the three forks in the design doc.

## Consequences

- **Positive:** decentralized execution as the thesis intends, with the property *gated* rather than
  asserted; no new dependencies; the M2 path stays intact as the comparison reference; the per-agent
  action becomes `Discrete(5)`, so off-policy learners become available later; transfer across K and
  communication-loss robustness become measurable for the first time.
- **Negative / trade-offs:** independent learners have no centralized critic, so credit assignment
  rests on the shared reward (mitigated by the recorded escalation); parameter sharing means one policy
  for all seats, not heterogeneous agents; the `AgentSplitVecEnv` transport is ours to maintain (~110
  lines) and must keep per-episode dashboard accounting on the team scale.
- **Neutral:** the EV stays scripted; the V2V channel stays lossless in training (dropout is evaluated,
  not trained against, until M4).
