# 0005. Phase the migration gym-first, then a ROS 2 mechanical demo

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

> **Superseded in part by [ADR 0011](0011-m4-ros2-mechanical-demo.md) (2026-09-04):** the *phase*
> decision below — gym-first, then a ROS 2 mechanical demo — **stands**. Only the route to it changes:
> the mechanical demo is *not* built on `f1tenth_gym_ros` (a Foxy/EOL bridge capped at 1–2 agents whose
> Humble branch would drag in a different f1tenth_gym and re-baseline every published table). M4 keeps
> that project's assets, topic conventions and launch pattern, vendored at a pinned SHA, and runs one
> `ClearanceEnv` as the only physics with K ROS 2 nodes — one per car.

## Context and problem statement

The v1.0.0 rewrite spans two concerns: the **RL research** (cooperative
multi-agent EV clearing) and a **ROS 2 mechanical/deployment demo**. Doing both at
once is slow and couples fast-moving research to heavier ROS 2 integration.

## Considered options

- **Gym-first:** build and validate the RL purely in `f1tenth_gym` (no ROS), then
  bridge to ROS 2 for a mechanical demo.
- **ROS 2-first:** stand up `f1tenth_gym_ros` (Humble) immediately and develop the
  RL inside the ROS loop.
- Both in parallel.

## Decision

We will go **gym-first**: implement the cooperative multi-agent EV-clearing RL in
`f1tenth_gym` headlessly (fast iteration, no ROS), then add a **ROS 2 Humble
mechanical demo** via `f1tenth_gym_ros` (1–2 cars), with a real car
(`f1tenth_system`) as an optional stretch. This mirrors the original thesis's split
(SUMO for RL results, Gazebo for the mechanical proof) — but on a maintained stack.

## Consequences

- **Positive:** fastest path to the research payoff; RL isn't blocked on ROS 2
  integration; clean separation of concerns.
- **Negative / trade-offs:** the stock `f1tenth_gym_ros` bridge exposes only 1–2
  agents, so the multi-car demo will need the bridge extended (or the demo scoped
  to a representative slice) when we reach that phase.
- **Neutral:** design scenarios with real headroom from day one (a lesson from the
  legacy toy harness, where a saturated task hid all algorithm differences).
