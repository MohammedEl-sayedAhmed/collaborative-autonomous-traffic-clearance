# 0004. Adopt f1tenth_gym as the RL platform

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

The legacy project ran reinforcement learning inside a heavy 2-car Gazebo scene
that OOMs on modest machines, and hand-rolled its own environment. We need an RL
platform that is maintained, fast, multi-agent capable (our problem is several
cars cooperating to clear a path for an emergency vehicle), and compatible with
modern RL libraries.

## Considered options

- **f1tenth_gym** `v1.0.0` branch — Gymnasium API, Python 3, natively N-agent
  (`num_agents`, per-agent ids), pygame rendering; MIT.
- f1tenth_gym `main` — the older `gym`-API version (Python 3.8/3.9); also N-agent.
- Keep the custom Gazebo RL environment.
- Our own headless kinematic harness (built for the legacy line) as the long-term base.

## Decision

We will build the RL on **`f1tenth_gym` (v1.0.0 branch)**, pinned to a specific
commit, and model our contribution **on top** of it: designate one of the N agents
as the **emergency vehicle**, add **V2V** as shared observation among the
cooperating agents, and a **cooperative reward** (minimize the EV's time-to-clear,
with anti-oscillation terms). The gym provides N-car physics, LiDAR, and collision
for free; the cooperative EV-clearing layer is our novel work.

## Considered but rejected reasoning

`main`'s old `gym` API is deprecated and won't plug cleanly into current libraries;
the custom Gazebo env and the kinematic harness are ours to maintain forever and
don't generalize. `v1.0.0` is a dev branch (no formal release) — accepted risk,
mitigated by pinning a commit.

## Consequences

- **Positive:** Gymnasium API drops straight into `stable-baselines3` (DQN/PPO);
  N-agent support makes the cooperative multi-agent scenario first-class; a
  maintained, cited platform instead of bespoke infrastructure.
- **Negative / trade-offs:** dependence on a pre-release branch (pin + watch for
  churn); the gym is a *racing* env, so EV semantics, V2V, and cooperation are ours
  to design.
- **Neutral:** our existing training dashboard is stack-agnostic (reads JSONL runs),
  so it carries over unchanged.
