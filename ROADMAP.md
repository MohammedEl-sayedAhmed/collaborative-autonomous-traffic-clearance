# Roadmap

Ideas for improving *Collaborative Autonomous Traffic Clearance*. The **project** comes first (the
learning, the simulation, the software stack), then **the dashboard**, where each item exists to show
or measure one of the project changes above it.

> Most of this list was written for the 2020 ROS 1 line. Where the v1.0.0 rewrite has already done an
> item, the note at the bottom says so.

Legend: ⚡ quick win · 🚀 bigger bet · 🐞 known bug (see [docs/legacy/KNOWN_ISSUES.md](docs/legacy/KNOWN_ISSUES.md))

---

## Part 1 — The project

### A. The learning itself (the biggest wins)
1. 🚀 **From a lookup table to a function.** The Q-table has about 1.4 million cells and most are
   never visited, so it cannot generalise between similar situations. Move to **tile coding / a linear
   model** (cheap) or a small **DQN**. The single biggest quality jump.
2. 🚀 **Wrap the environment as a `gymnasium` environment.** The small headless simulator already has
   the exact state, actions and reward. Expose it as a standard Gym environment so
   **stable-baselines3** (DQN, PPO) can train on it. Little glue code needed.
3. ⚡ **A better reward.** Today it is only the ambulance's change in speed (never positive). Add
   terms for the car's own progress, a small penalty per lane change, a bonus for keeping a safe gap,
   and a big bonus at the end for a fast clearance.
4. ⚡ **Richer state:** add "is the left / right lane free?" worked out from V2V, so the car can
   reason directly about where to go.
5. ⚡ **Exploration:** softmax (Boltzmann) instead of ε-greedy; add experience replay.

### B. From one car to the real "collaborative" idea
6. 🚀 **Several cars learning together.** The 2020 project trains one car. Extend it to several cars
   learning to open a corridor together (independent learners first, then methods like QMIX or
   MADDPG). The headline future work.
7. 🚀 **Share intentions over V2V.** Broadcast "I am changing to the left lane", not just where the
   car is, so the others can plan around each other's *plans*.

### C. Environment and scenarios
8. 🐞 **Stop jinja2 overwriting** launch files that are under version control at run time (write the
   generated files to a separate folder).
9. ⚡ **Take the road geometry out of the code** (lane limits, x range and lane count are hard-coded
   for `threeLanes`) and put it in a config, so curves, more lanes and denser traffic become possible.
10. 🚀 **Generated scenarios plus randomisation** (traffic density, car dynamics, sensor noise) for
    robustness, plus a **benchmark set** (light / heavy / curve / several EVs) with a scorecard.
11. ⚡ **Gazebo faster than real time** (`real_time_update_rate`), Gazebo without a screen, and
    several simulations in parallel, so real training becomes practical on a good machine.
    M5 replaces this: Gazebo Harmonic, headless, stepped in lockstep by the bridge (ADR 0013).

### D. Navigation and control
12. 🐞 **Wrong TEB `local_plan` topic:** the "which lane will I be in next" detection listens on the
    wrong topic and almost never fires (`threeLanes_current_future_pos.py`).
13. 🐞 **`move_base_follower` units** (an angular velocity is used as a steering angle) and the
    **5 m goal tolerance**.
14. 🚀 **A learned local planner:** let the learner replace the hand-tuned TEB planner for the
    move-aside manoeuvre.

### E. Engineering and reproducibility
15. 🚀 **Get off ROS Kinetic / Python 2** (both unsupported): move the learning to **Python 3**, then
    the stack to **Noetic** or **ROS 2**.
16. ⚡ **CI** (GitHub Actions): build the image, run the headless smoke test and the Playwright
    dashboard test on every push (both tests already exist).
17. ⚡ **Unit tests** for the state binning, the Q-update, and how V2V messages are combined.
18. **Be honest about sim-to-real:** training uses perfect position data. Add AMCL or noise to close
    the gap.
    Planned as M5.2: simulated lidar plus AMCL per car, the localization error published (ADR 0013).

### F. Realistic V2V
19. ⚡ Fix the **range check that is not applied everywhere** (car footprints are written regardless
    of `comm_range`).
20. 🚀 Model a real radio: lost packets, delay, limited bandwidth, and cars joining and leaving beyond
    the hard limit of 6 cars. M4.4 measured why this matters (design doc, M4.4 results): on HARD a lost
    or 500 ms old broadcast makes the occupied lane look empty and the cars collide, while on STRICT the
    policy never needed the radio. A real fix keeps the last heard position for a while, treats silence
    as "not clear", and trains with loss and delay in the loop.

---

## Part 2 — The dashboard

Each item here supports one project change above.

| Dashboard work | Supports |
|----------------|----------|
| 🚀 **Q-table / policy heatmap** (best action per state) | A1, A3: see *what* the car learned |
| 🚀 **Per-agent curves + cooperation metrics** (zipper score, min inter-car gap) | B6 multi-agent |
| ⚡ **Runs over several seeds, with mean ± std bands** | A1–A5, C10: a proper comparison, not one noisy curve |
| 🚀 **DQN/loss panels** (TD-loss, mean Q, replay stats) once function approximation lands | A1, A2 |
| 🚀 **Episode replay**: a top-down animation of the cars you can scrub through | C, D: see the behaviour, not just curves |
| ⚡ **Hyperparameter sweep view**: run many headless runs, plot the trade-off front | A, C |
| ⚡ **More logged metrics** (lane changes, min gap, time-to-clear, near-misses) | C10 benchmark scorecard |
| ⚡ **Benchmark scorecard tab** (per-scenario results) | C10 |
| ⚡ CSV / PNG export; remember the selected runs and zoom in `localStorage`; a report generator | all: sharing results |

---

## The top 3 by value for effort
1. **Gymnasium wrapper + a real learner**: turns the toy into learning that generalises. Dashboard:
   loss / Q panels and multi-seed bands.
2. **Several cars**: delivers the actual collaborative thesis. Dashboard: per-car curves and
   cooperation metrics.
3. **A better reward + configurable scenarios**: makes the learning meaningful and general. Dashboard:
   benchmark scorecard and episode replay.

> **Where the v1.0.0 line stands (2026-09):** items 1, 2, 3, 4, 6, 15 and 17 are done in `caatc/` on
> `f1tenth_gym`. M1 built the Gymnasium environment with a multi-car action, a shared reward and the
> "is the side lane free" features. M2 trained it with PPO. M3 made each car decide alone (item 6) and
> added the range check (item 19) and message-loss evaluation (part of 20). M4 moved it onto ROS 2
> (item 15) and measured message loss and delay through a relay on the real bus (the measuring half
> of 20). `master` is tagged `v1.0.0`. Next is M5 (ADR 0013): a 3D plant in Gazebo Harmonic behind the
> same seam (items 11 and 18), then one real car in the loop. Still open: intention sharing (7), generated scenarios and a benchmark set (10),
> CI (16), a realistic radio (20), and most of the dashboard items. The old headless simulator that
> tried this on the ROS 1 line is at tag `v0.3.0`, `tools/rl_harness/train.py`.
