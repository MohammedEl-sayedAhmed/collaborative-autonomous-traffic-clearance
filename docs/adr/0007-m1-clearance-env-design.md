# 0007. M1 scenario — ACC-governed multi-lane move-aside (ClearanceEnv)

- **Status:** proposed
- **Date:** 2026-08-25
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

M1 must turn the bare `f1tenth_gym` base (M0) into our actual problem — several cars cooperating to
clear a path for an emergency vehicle — as a headless Gymnasium wrapper. `f1tenth_gym` is a *racetrack*
env with no EV, cooperation, or V2V, so we design those on top. The load-bearing question is **how to
enforce "blocking"** such that a naive policy is clearly worse than a cooperative one *by construction*
(the legacy toy harness taught us that a saturated scenario, where even random ≈ optimal, hides every
algorithm difference).

## Considered options

- **ACC on wide virtual lanes** — cooperators cruise slow; a scripted EV runs an adaptive-cruise law, so
  a car left in its lane clamps it to convoy speed (graded slowness, no crash) and a move-aside lets it
  sprint. Headroom = a robust ~4× speed ratio.
- **Narrow physical corridor** — a tight lane requiring cm-precise sub-lane tucking. Rejected: risks
  *both* saturation failure modes (if weak low-speed steering can't execute the tuck, a good policy also
  fails), and reworks the EV controller + action space (continuous offset).
- **Racetrack overtaking** / other mappings — less faithful to the cooperative move-aside story.

## Decision

Adopt the **ACC-governed multi-lane move-aside** design: `ClearanceEnv` wrapper, scripted EV at
`agent_0`, K trained cooperators on a virtual 3-lane road, `Discrete(5)` move-aside actions, V2V shared
observation, a shared cooperative reward, and **EASY** (timing-only) as the first trainable preset
before **HARD** (occupancy reasoning). Headroom is proven by a pre-training gate before any learning is
spent. Full design and implementation plan: [`../design/m1-clearance-env.md`](../design/m1-clearance-env.md).

Status is **proposed** pending the owner's confirmation of the two forks noted in the design doc.

## Consequences

- **Positive:** headroom is guaranteed by geometry + the ACC speed ratio, not by fragile steering;
  staggered cooperators give a dense (monotone) learning gradient; API-accurate to the pinned gym
  commit; centralized-SB3-ready now, CTDE-ready later; the existing dashboard visualizes it unchanged.
- **Negative / trade-offs:** the EV is scripted (not learned) in M1; the "road" is virtual (lanes are
  lateral offsets, not real map lanes); the HARD preset and V2V occupancy features add complexity.
- **Neutral:** M1 also removes the legacy ROS 1 tree (ADR 0003) once the gym scenario is in.
