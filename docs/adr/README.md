# Architecture Decision Records

This folder holds the **big decisions** made in this project, one short file each, in the
[MADR](https://adr.github.io/madr/) style. The point is to keep the *why* behind each decision,
especially while the project moves from its old ROS 1 / Python 2 stack to a supported
ROS 2 / Python 3 one.

## Rules

- One file per decision: `NNNN-kebab-case-title.md`. Numbers go up by one and are never reused.
- **Status:** `proposed` → `accepted` → later `superseded by NNNN` or `deprecated`.
- An accepted ADR is **not changed** to say something different. To change a decision, write a new ADR
  that supersedes the old one, then update the old one's status and link to the new one. Rewording an
  ADR so it is easier to read is fine, as long as the decision itself stays the same.
- Start a new one by copying [`0000-template.md`](0000-template.md).

## Index

| # | Title | Status |
|---|-------|--------|
| [0001](0001-record-architecture-decisions.md) | Write decisions down as ADRs | Accepted |
| [0002](0002-migrate-to-ros2-humble-python3.md) | Move from ROS Kinetic / Python 2 to ROS 2 / Python 3 | Accepted *(distro superseded by [0012](0012-target-ros2-jazzy-not-humble.md))* |
| [0003](0003-refactor-in-place-preserve-legacy-with-tags.md) | Rebuild in this repo; keep the old stack at version tags | Accepted |
| [0004](0004-adopt-f1tenth-gym-for-rl.md) | Use f1tenth_gym as the simulator for learning | Accepted |
| [0005](0005-phase-migration-gym-first.md) | Do the learning in the plain simulator first, then a ROS 2 demo | Accepted *(bridge part superseded by [0011](0011-m4-ros2-mechanical-demo.md))* |
| [0006](0006-keep-thesis-as-private-submodule.md) | Keep the thesis in a private submodule | Accepted |
| [0007](0007-m1-clearance-env-design.md) | M1 scenario: the EV uses adaptive cruise, on wide lanes | Accepted |
| [0008](0008-train-with-stable-baselines3-ppo.md) | M2: train with stable-baselines3 PPO | Accepted |
| [0009](0009-decentralized-execution-ippo.md) | M3: each car decides alone, with one shared policy (IPPO) | Accepted |
| [0010](0010-strict-preset-removes-the-convoying-substitution.md) | The STRICT preset: speeding up must not count as cooperation | Accepted |
| [0011](0011-m4-ros2-mechanical-demo.md) | M4: the ROS 2 Jazzy demo, one simulator plus K car nodes | Accepted, built |
| [0012](0012-target-ros2-jazzy-not-humble.md) | Target ROS 2 Jazzy, not Humble (supported to 2029) | Accepted |
| [0013](0013-m5-3d-plant-gazebo-harmonic.md) | M5: a 3D plant in Gazebo Harmonic behind the same seam, then one real car in the loop | Accepted, not started |
