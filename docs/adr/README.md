# Architecture Decision Records

This directory records the **significant architectural decisions** for this
project as lightweight [MADR](https://adr.github.io/madr/)-style records — so the
*why* behind each decision is captured, especially as the project is refactored
from its legacy ROS 1 / Python 2 stack to a maintained ROS 2 / Python 3 one.

## Conventions

- One file per decision: `NNNN-kebab-case-title.md`, zero-padded and **sequential**;
  numbers are never reused or renumbered.
- **Status:** `proposed` → `accepted` → (later) `superseded by NNNN` / `deprecated`.
- ADRs are **immutable** once accepted: don't rewrite one to change the decision —
  add a new ADR that supersedes it, and update the old one's status + a link.
- Start a new one by copying [`0000-template.md`](0000-template.md).

## Index

| # | Title | Status |
|---|-------|--------|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions in ADRs | Accepted |
| [0002](0002-migrate-to-ros2-humble-python3.md) | Migrate from ROS Kinetic / Python 2 to ROS 2 Humble / Python 3 | Accepted |
| [0003](0003-refactor-in-place-preserve-legacy-with-tags.md) | Refactor in place; preserve the legacy stack via SemVer tags | Accepted |
| [0004](0004-adopt-f1tenth-gym-for-rl.md) | Adopt f1tenth_gym as the RL platform | Accepted |
| [0005](0005-phase-migration-gym-first.md) | Phase the migration gym-first, then a ROS 2 mechanical demo | Accepted |
| [0006](0006-keep-thesis-as-private-submodule.md) | Keep the thesis as a private git submodule | Accepted |
