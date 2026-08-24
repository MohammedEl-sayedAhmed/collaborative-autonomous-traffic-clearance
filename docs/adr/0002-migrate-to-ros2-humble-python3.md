# 0002. Migrate from ROS Kinetic / Python 2 to ROS 2 Humble / Python 3

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

The project runs on **ROS Kinetic (EOL 2021), Gazebo 7, and Python 2 (EOL 2020)**.
These can't be installed on a modern distro without heavy containerization, get no
security/bug fixes, and block use of current libraries and the current F1TENTH
tooling. The F1TENTH org itself has archived its ROS 1 pieces (`f1tenth_simulator`,
`f110_ros`) and moved to ROS 2.

## Considered options

- **ROS 2 Humble** (LTS, supported to 2027).
- **ROS 2 Foxy** — what the current F1TENTH bridge/system repos target, but **EOL**
  (May 2023).
- Stay on ROS 1 Kinetic (containerized) indefinitely.
- ROS 2 Rolling / newer non-LTS.

## Decision

We will migrate to **ROS 2 Humble** and **Python 3**. Humble is the current LTS;
targeting Foxy would repeat the very EOL problem we are escaping, and the F1TENTH
gym already has a `dev-humble` branch to build from.

## Consequences

- **Positive:** a supported, modern stack; access to current libraries and the
  maintained F1TENTH ecosystem; a real foundation for further research.
- **Negative / trade-offs:** the F1TENTH `*_ros`/`system` repos default to Foxy, so
  some porting to Humble is required; ROS 2 is a substantial paradigm shift from
  ROS 1 (rclpy, launch, DDS) — effectively a rewrite, not a port.
- **Neutral:** the legacy ROS 1 code remains reachable (see ADR 0003).
