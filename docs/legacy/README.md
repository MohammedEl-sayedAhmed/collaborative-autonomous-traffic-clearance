# Legacy documentation — the 2020 ROS 1 / Python 2 stack

These documents describe the **original 2020 graduation project** built on
**ROS Kinetic / Gazebo 7 / Python 2** (the F1TENTH / MIT `racecar-simulator`
stack). That code was removed from `master` in milestone **M1** of the ROS 2 /
Python 3 migration (see [ADR 0003](../adr/0003-refactor-in-place-preserve-legacy-with-tags.md)),
but it is **preserved, buildable, at the annotated tags**:

| Tag | What it is |
|-----|------------|
| `v0.1.0` | Baseline — the original project as submitted (blocker bugs intact) |
| `v0.2.0` | Reproducible & core defects fixed; containerized runner + dashboard |
| `v0.3.0` | Enhanced — final ROS 1 / Python 2 line |

To run or read the legacy stack, check out a tag — the code *and* these docs
apply to it there:

```bash
git checkout v0.3.0     # the full ROS 1 / Gazebo project + its run.sh sim|nav|rl|…
```

| Doc | Contents (legacy stack) |
|-----|-------------------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layered architecture, system data-flow, one end-to-end episode, the RL loop |
| [SUBSYSTEMS.md](SUBSYSTEMS.md) | Each layer (simulation, V2V, navigation, move_car, control, RL) |
| [PACKAGES.md](PACKAGES.md) | All 36 ROS packages + every custom message / service / action |
| [RUNNING.md](RUNNING.md) | Containerized setup, every Gazebo demo, the RL run order, troubleshooting |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Verified issues, applied fixes, behaviour-changing fixes left for review |
| [RL_EXPERIMENTS.md](RL_EXPERIMENTS.md) | The fix-by-fix RL campaign (legacy headless harness + real Gazebo) |

The **current** (v1.0.0, ROS 2 / Python 3) work lives on `master`: the `caatc/`
package on `f1tenth_gym`, the architecture decisions in [`../adr/`](../adr/), and
the M1 design in [`../design/`](../design/). The **training dashboard**
([`../DASHBOARD.md`](../DASHBOARD.md)) is stack-agnostic and carries over unchanged.
