# 0012. Target ROS 2 Jazzy, not Humble

- **Status:** accepted
- **Date:** 2026-09-04
- **Deciders:** Mohammed El-sayed Ahmed
- **Supersedes:** only the *release* chosen in [ADR 0002](0002-migrate-to-ros2-humble-python3.md)
  (Humble → Jazzy). The rest of ADR 0002, leaving ROS Kinetic / Python 2 for a supported ROS 2 release
  and Python 3, stands unchanged.

## The problem

[ADR 0002](0002-migrate-to-ros2-humble-python3.md) chose **ROS 2 Humble** in August 2026 because it was
"the current long-term release, supported to 2027". The whole point of the v1.0.0 rewrite was to stop
depending on software that is no longer supported (ROS Kinetic, Ubuntu 16.04, Python 2).

Two things changed by the time M4 actually needed a ROS release:

1. **Humble is almost out of support.** Its support ends in **May 2027**, about **eight months** from
   today (2026-09-04). Writing the project's first ROS 2 code on it would mean adopting a stack that dies
   within the year, exactly the situation ADR 0002 was meant to escape.
2. **Jazzy Jalisco is out and is the current long-term release**, supported until **May 2029**.

I checked this by pulling the images rather than trusting memory:

| release | Ubuntu | Python | support ends |
|---------|--------|--------|--------------|
| `ros:humble` | 22.04 | **3.10** | May 2027 (about 8 months) |
| `ros:jazzy` | 24.04.4 LTS | **3.12.3** | **May 2029** |

The Python version matters beyond packaging: `rclpy` is compiled against the release's Python, and
[ADR 0011](0011-m4-ros2-mechanical-demo.md) commits to **one Python across the whole stack**, so the
physics is identical in the ROS image and in the headless one. So the ROS release decides which Python
our own images must use.

## Options

1. **Jazzy.** Supported to 2029, Python 3.12. Means moving `caatc-gym` (and the images built on it)
   from Python 3.11 to 3.12, and re-checking every published table on the new Python. **Chosen.**
2. **Humble, as ADR 0002 says.** The best documented release, and Python 3.10 is a smaller step from
   3.11. Rejected: it is unsupported in about 8 months, so the demo would ship onto a dead stack and need
   redoing right away. I want no end-of-life dependencies in this project.
3. **Kilted Kaiju.** Newer, but not a long-term release, with a short support window. Rejected: it is
   supported for less time than Humble.
4. **Rolling.** Always the latest, never stable. Not suitable for a demo whose numbers are published.

Whether option 1 was even possible was checked before deciding, not assumed: on Python 3.12 the whole
stack installs (`numpy` 2.5.2, `numba` 0.67.0, `gymnasium` 0.29.1, `opencv-python`, `shapely`,
`scipy`), and the pinned `f1tenth_gym@v1.0.0` declares `python = ">=3.9"` with no upper limit.

## Decision

Target **ROS 2 Jazzy** for M4 and after, and move the project's images to **Python 3.12** to match.
Keep the ROS release a **parameter** of the ROS image (`ARG ROS_DISTRO=jazzy`), so the next long-term
release is a one-line change that can be tried on purpose rather than found out by accident.

Every published table is re-checked on the new Python **before** any ROS code is written. Any change is
re-measured and written down, never quietly absorbed.

## Consequences

- **Good:** about 2.7 years of support instead of 8 months; a newer Python; Ubuntu 24.04 instead of
  22.04; and the project stops taking on the exact problem it was migrating away from. One Python
  everywhere keeps the physics identical by construction.
- **Cost:** Humble has more tutorials and more third-party packages pinned to it, so some ROS code will
  need newer equivalents. Moving to Python 3.12 meant re-checking the M1 headroom test and every M2 and
  M3 number, and `numpy` moved from 2.4.6 to 2.5.2 in the process. (Done: 16 of 17 rows reproduce
  exactly; one model had to be retrained, see ADR 0011.) Any package that exists only for Humble is out
  of scope by construction, which is the intended effect.
- **Neutral:** the rest of ADR 0002 is untouched; only its release changes. ADR 0011's design does not
  depend on the release (one simulator, K car nodes, standard messages), so it only needed renaming.
