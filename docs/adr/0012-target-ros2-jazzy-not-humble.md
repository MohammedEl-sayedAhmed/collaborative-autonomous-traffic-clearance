# 0012. Target ROS 2 Jazzy, not Humble

- **Status:** accepted
- **Date:** 2026-09-04
- **Deciders:** Mohammed El-sayed Ahmed
- **Supersedes:** the *distro* choice in [ADR 0002](0002-migrate-to-ros2-humble-python3.md) (Humble →
  Jazzy). ADR 0002's substance — leave ROS Kinetic / Python 2, adopt a supported ROS 2 LTS and
  Python 3 — stands unchanged.

## Context and problem statement

[ADR 0002](0002-migrate-to-ros2-humble-python3.md) chose **ROS 2 Humble** in August 2026, on the
grounds that it was "the current LTS, supported to 2027". The entire point of the v1.0.0 migration was
to stop depending on an end-of-life stack (ROS Kinetic / Ubuntu 16.04 / Python 2).

Two things have changed by the time M4 actually needs a distro:

1. **Humble is nearly expired.** Its EOL is **May 2027** — about **eight months** from today
   (2026-09-04). Standing up the project's first ROS 2 code on it would mean adopting a stack that
   dies within the year, which is precisely the situation ADR 0002 existed to escape.
2. **Jazzy Jalisco is available and is the current LTS**, supported to **May 2029**.

Verified empirically rather than from memory, by pulling the images:

| distro | Ubuntu | Python | support ends |
|--------|--------|--------|--------------|
| `ros:humble` | 22.04 | **3.10** | May 2027 (~8 months) |
| `ros:jazzy` | 24.04.4 LTS | **3.12.3** | **May 2029** |

The interpreter matters beyond packaging: `rclpy` is built against the distro's Python, and
[ADR 0011](0011-m4-ros2-mechanical-demo.md) commits to running **one interpreter across the whole
stack** so the physics is identical in the ROS image and the headless one. So the distro choice fixes
the interpreter our own images must use.

## Considered options

1. **Jazzy** — LTS to 2029, Python 3.12. Requires rebasing `caatc-gym` (and the images built on it)
   from Python 3.11 to 3.12, and re-verifying every published table on the new interpreter. **Chosen.**
2. **Humble, as ADR 0002 says** — the most documented distro, and Python 3.10 is a smaller jump from
   3.11. Rejected: it goes EOL in ~8 months, so the "mechanical demo" milestone would ship onto a dead
   stack and need redoing immediately. The owner's instruction was explicit: *no EOL dependencies.*
3. **Kilted Kaiju** — newer, but a non-LTS release with a short support window. Rejected: shorter
   support than Humble.
4. **Rolling** — always-current, never stable; unsuitable for a reproducible demo whose numbers are
   published.

Feasibility of option 1 was checked before deciding, not assumed: on Python 3.12 the resolver finds
cp312 wheels for the whole stack — `numpy` 2.5.2, `numba` 0.67.0, `gymnasium` 0.29.1, `opencv-python`
(abi3), `shapely`, `scipy` — and the pinned `f1tenth_gym@v1.0.0` declares `python = ">=3.9"` with no
upper bound.

## Decision

Target **ROS 2 Jazzy** for M4 and beyond, and rebase the project's images on **Python 3.12** to match
its interpreter. Keep the ROS image's distro **parametric** (`ARG ROS_DISTRO=jazzy`) so the next LTS is
a one-line change that can be exercised deliberately rather than discovered.

Every published table is re-verified on the new interpreter **before** any ROS code is written; drift,
if any, is re-baselined and documented, never silently absorbed.

## Consequences

- **Positive:** ~2.7 years of support instead of ~8 months; a newer Python; Ubuntu 24.04 rather than
  22.04; and the project stops re-acquiring the exact liability it was migrating away from. One
  interpreter across headless and ROS images keeps the physics identical by construction.
- **Negative / trade-offs:** Humble has more tutorials and more third-party packages pinned to it, so
  some ROS ecosystem code will need newer equivalents. Rebasing to Python 3.12 means re-verifying the
  M1 gate and every M2/M3 number, and `numpy` moves 2.4.6 → 2.5.2 in the process, so a re-baseline is
  possible. Any package that exists only for Humble is out of scope by construction — which is the
  intended effect.
- **Neutral:** ADR 0002's substance is untouched; only its distro row changes. ADR 0011's design is
  distro-agnostic — one plant, K car nodes, standard messages — so it needed only renaming.
