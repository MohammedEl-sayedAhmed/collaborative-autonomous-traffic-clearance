# 0002. Move from ROS Kinetic / Python 2 to ROS 2 Humble / Python 3

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

> **The ROS version was changed by [ADR 0012](0012-target-ros2-jazzy-not-humble.md) (2026-09-04).**
> The decision to leave ROS Kinetic / Python 2 for a supported ROS 2 release and Python 3 **stands**.
> Only the release changes: **Jazzy** (Ubuntu 24.04, Python 3.12, supported to May 2029) replaces
> **Humble**, whose support ends in May 2027, about eight months after our first ROS 2 code would have
> shipped on it.

## The problem

The project runs on **ROS Kinetic (unsupported since 2021), Gazebo 7, and Python 2 (unsupported since
2020)**. These cannot be installed on a modern system without heavy containers, get no security or bug
fixes, and block us from using current libraries and the current F1TENTH tools. The F1TENTH group
itself has archived its ROS 1 parts (`f1tenth_simulator`, `f110_ros`) and moved to ROS 2.

## Options

- **ROS 2 Humble** (long-term support release, supported to 2027).
- **ROS 2 Foxy**. This is what the current F1TENTH bridge and system repos target, but it is
  **unsupported** since May 2023.
- Stay on ROS 1 Kinetic (in a container) forever.
- ROS 2 Rolling, or a newer non-LTS release.

## Decision

We move to **ROS 2 Humble** and **Python 3**. Humble is the current long-term release. Choosing Foxy
would repeat the exact problem we are trying to escape. The F1TENTH gym already has a `dev-humble`
branch to build from.

## Consequences

- **Good:** a supported, modern stack. We can use current libraries and the maintained F1TENTH
  tools. A real base for further work.
- **Cost:** the F1TENTH `*_ros` / `system` repos default to Foxy, so some porting is needed. ROS 2 is
  a big change from ROS 1 (rclpy, launch files, DDS). In practice this is a rewrite, not a port.
- **Neutral:** the old ROS 1 code stays reachable (see ADR 0003).
