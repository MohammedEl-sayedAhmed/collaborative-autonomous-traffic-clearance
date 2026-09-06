# 0005. Do the learning in the plain simulator first, then a ROS 2 demo

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

> **Partly superseded by [ADR 0011](0011-m4-ros2-mechanical-demo.md) (2026-09-04).** The *order*
> decided below, learning first and a ROS 2 demo after, **stands**. Only the way to build the demo
> changes. It is *not* built on `f1tenth_gym_ros`, a Foxy-era bridge that supports only 1 or 2 cars and
> whose Humble branch would pull in a different f1tenth_gym, which would change every published number.
> M4 keeps only that project's topic names (nothing is copied from it) and runs one `ClearanceEnv` as
> the only physics, with K ROS 2 nodes, one per car.

## The problem

The v1.0.0 rewrite has two parts: the **learning research** (several cars cooperating to clear the
EV's path) and a **ROS 2 demo** that shows it running the way a real robot would. Doing both at once
is slow, and it ties fast-moving research work to heavier ROS 2 integration.

## Options

- **Learning first:** build and check the learning purely in `f1tenth_gym` (no ROS), then connect it
  to ROS 2 for a demo.
- **ROS 2 first:** set up `f1tenth_gym_ros` (Humble) right away and develop the learning inside the
  ROS loop.
- Both at the same time.

## Decision

We go **learning first**: build the cooperative EV-clearing learning in `f1tenth_gym` without a screen
(fast to iterate, no ROS), then add a **ROS 2 Humble demo** via `f1tenth_gym_ros` (1 or 2 cars), with
a real car (`f1tenth_system`) as an optional extra. This mirrors the split in the original thesis
(SUMO for the learning results, Gazebo for the mechanical proof), but on a supported stack.

## Consequences

- **Good:** the fastest way to the research result. The learning is not blocked on ROS 2 work. The
  two concerns stay separate.
- **Cost:** the stock `f1tenth_gym_ros` bridge only exposes 1 or 2 cars, so the multi-car demo will
  need the bridge extended (or the demo limited to a slice) when we get there.
- **Neutral:** design scenarios with real headroom from day one. In the old project the task was so
  easy that every algorithm looked the same.
