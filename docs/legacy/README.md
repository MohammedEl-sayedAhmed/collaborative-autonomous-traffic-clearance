# Legacy documentation: the 2020 ROS 1 / Python 2 stack

These documents describe the **original 2020 graduation project**, built on
**ROS Kinetic / Gazebo 7 / Python 2** (the F1TENTH / MIT `racecar-simulator`
stack). That code was removed from `master` in milestone **M1** of the move to
ROS 2 / Python 3 (see [ADR 0003](../adr/0003-refactor-in-place-preserve-legacy-with-tags.md)),
but it is **kept, and still builds, at these tags**:

| Tag | What it is |
|-----|------------|
| `v0.1.0` | The original project as submitted (with its blocking bugs) |
| `v0.2.0` | Reproducible, main bugs fixed; Docker runner and dashboard added |
| `v0.3.0` | Improved; the last ROS 1 / Python 2 version |

To run or read the old stack, check out a tag. The code *and* these docs apply
to it there:

```bash
git checkout v0.3.0     # the full ROS 1 / Gazebo project + its run.sh sim|nav|rl|…
```

| Doc | Contents (legacy stack) |
|-----|-------------------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The layers, how data flows, one full episode, the learning loop |
| [SUBSYSTEMS.md](SUBSYSTEMS.md) | Each layer (simulation, V2V, navigation, move_car, control, learning) |
| [PACKAGES.md](PACKAGES.md) | All 36 ROS packages and every custom message / service / action |
| [RUNNING.md](RUNNING.md) | Docker setup, every Gazebo demo, how to run the learning, troubleshooting |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Confirmed problems, applied fixes, and fixes left for review |
| [RL_EXPERIMENTS.md](RL_EXPERIMENTS.md) | The fix-by-fix learning campaign (a small headless simulator and real Gazebo) |

The **current** work (v1.0.0, ROS 2 / Python 3) lives on `master`: the `caatc/`
package on `f1tenth_gym`, the decisions in [`../adr/`](../adr/), and the designs
in [`../design/`](../design/). The **training dashboard**
([`../DASHBOARD.md`](../DASHBOARD.md)) only reads JSONL files, so it works for both.
