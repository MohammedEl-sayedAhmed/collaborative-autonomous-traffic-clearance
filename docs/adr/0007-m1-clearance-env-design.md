# 0007. M1 scenario: the EV uses adaptive cruise, on wide lanes (ClearanceEnv)

- **Status:** accepted
- **Date:** 2026-08-25 (accepted 2026-08-26)
- **Deciders:** Mohammed El-sayed Ahmed

## The problem

M1 has to turn the bare `f1tenth_gym` simulator (M0) into our real problem: several cars working
together to clear a path for an emergency vehicle (EV), as a Gymnasium environment that runs without a
screen. `f1tenth_gym` is a *race track* simulator. It has no EV, no cooperation and no V2V, so we design
those on top.

The key question is **how "blocking" should work**. We want a policy that does nothing to be clearly
worse than one that cooperates, *by design*. The 2020 project taught us why: its task was so easy that
even random actions looked optimal, so no algorithm difference could show.

## Options

- **Adaptive cruise (ACC) on wide virtual lanes.** Our cars drive slowly. The scripted EV follows
  whatever is in front of it at a safe distance, like adaptive cruise control in a modern car. So a car
  left in the EV's lane holds it down to convoy speed (slow, but no crash), and a car that moves aside
  lets it sprint. The gap is about a 4× difference in speed.
- **A narrow physical corridor.** A tight lane where a car has to tuck into a sub-lane with
  centimetre precision. Rejected: if the low-speed steering cannot do the tuck, then a good policy fails
  too, so the task can look impossible or trivial for the wrong reasons. It would also need a new EV
  controller and a continuous action (an offset), not a simple choice.
- **Race-track overtaking**, or other mappings. Less faithful to the "move aside for the ambulance"
  story.

## Decision

Use the **adaptive-cruise, wide-lane move-aside** design: a `ClearanceEnv` wrapper, a scripted EV at
`agent_0`, K learned cars on a virtual 3-lane road, five move-aside actions per car (`Discrete(5)`),
V2V information in each car's observation, one shared reward, and **EASY** (timing only) as the first
preset to train on, before **HARD** (which also needs the cars to read which side lane is free). A
check before any training proves the headroom is there. Full design and plan:
[`../design/m1-clearance-env.md`](../design/m1-clearance-env.md).

**Both open questions were decided on 2026-08-26:** (1) blocking = **ACC on wide lanes**, not a
narrow corridor; (2) first preset = **EASY**, not HARD. Status moved to **accepted**.

## Consequences

- **Good:** the headroom comes from the road geometry and the ACC speed ratio, not from delicate
  steering. Cars spread along the road give a reward that grows with each car that moves aside in
  time. The code matches the pinned gym commit exactly. It works with a central stable-baselines3
  learner now, and can be split per car later. The existing dashboard shows it without changes.
- **Cost:** the EV is scripted, not learned, in M1. The "road" is virtual (lanes are offsets from a
  centre line, not lanes on a map). The HARD preset and the V2V "which lane is free" features add
  complexity.
- **Neutral:** M1 also removes the old ROS 1 tree (ADR 0003) once the gym scenario is in.
