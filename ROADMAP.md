# Roadmap

Improvement ideas for *Collaborative Autonomous Traffic Clearance*, ordered **project first**
(the RL system, simulation, and stack) and then **the dashboard accordingly** — each dashboard item
exists to visualize or measure a specific project change.

Legend: ⚡ quick win · 🚀 bigger bet · 🐞 known bug (see [docs/legacy/KNOWN_ISSUES.md](docs/legacy/KNOWN_ISSUES.md))

---

## Part 1 — The project

### A. The RL brain (biggest levers)
1. 🚀 **Tabular → function approximation.** The Q-table is ~1.4 M cells and mostly unvisited, so it
   never generalizes across similar states. Move to **tile-coding / linear** (cheap) or a small
   **DQN**. Single biggest quality jump.
2. 🚀 **Wrap the env as a `gymnasium` environment.** The headless harness already models the exact
   state/action/reward — expose it as a standard Gym env so **stable-baselines3** (DQN/PPO) can train
   it. Bridges toy harness → real RL with little glue.
3. ⚡ **Reward shaping.** Today: only ambulance Δvelocity (≤0). Add potential-based terms — agent
   progress, small per-lane-change penalty, safety-margin bonus, big terminal bonus for fast
   clearance.
4. ⚡ **Enrich the state** with "is the left/right lane clear?" derived from V2V, so the agent can
   directly reason about escape lanes.
5. ⚡ **Exploration**: Boltzmann/softmax instead of ε-greedy; add experience replay.

### B. From single-agent to the real "collaborative" vision
6. 🚀 **Multi-agent RL.** The project trains one car; extend to several co-learning to open a
   corridor (independent Q-learners → CTDE methods like QMIX/MADDPG). The headline future work.
7. 🚀 **Intention sharing over V2V** — broadcast "changing left", not just footprints, so peers plan
   around each other's *plans*.

### C. Environment & scenarios
8. 🐞 **Fix jinja2 clobbering** version-controlled launch files at runtime (render to a generated dir).
9. ⚡ **De-hardcode geometry** (lane thresholds, x-range, lane count are baked to `threeLanes`) →
   config-drive it for curves / more lanes / denser traffic.
10. 🚀 **Procedural scenarios + domain randomization** (traffic density, dynamics, sensor noise) →
    robustness, plus a **benchmark suite** (light / heavy / curve / multi-EV) with a scorecard.
11. ⚡ **Faster-than-real-time Gazebo** (`real_time_update_rate`) + headless `gzserver` + parallel sim
    instances → makes real training practical on a capable machine.

### D. Navigation & control
12. 🐞 **TEB `local_plan` topic mismatch** — future-lane detection subscribes to the wrong topic and
    rarely fires (`threeLanes_current_future_pos.py`).
13. 🐞 **`move_base_follower` units** (angular velocity mapped to steering angle) and the **5 m goal
    tolerance**.
14. 🚀 **Learned local planner** — let RL replace hand-tuned TEB for the move-aside maneuver.

### E. Engineering & reproducibility
15. 🚀 **Escape ROS Kinetic / Python 2** (both EOL): port the RL to **Py3**, then the stack to
    **Noetic** or **ROS 2**.
16. ⚡ **CI** (GitHub Actions): build the image + run the headless-harness smoke test + the Playwright
    dashboard test on push (both tests already exist).
17. ⚡ **Unit tests** for state discretization, the Q-update, and V2V aggregation.
18. **Sim-to-real honesty:** training uses ground-truth odometry — add AMCL/noise to close the gap.

### F. V2V realism
19. ⚡ Fix the **range-gating inconsistency** (footprints written regardless of `comm_range`).
20. 🚀 Model realistic comms: packet loss, latency, bandwidth, dynamic membership beyond the hard cap
    of 6 cars.

---

## Part 2 — The dashboard (accordingly)

Each item supports a project change above.

| Dashboard work | Supports |
|----------------|----------|
| 🚀 **Q-table / policy heatmap** (best action per state) | A1, A3 — see *what* the agent learned |
| 🚀 **Per-agent curves + cooperation metrics** (zipper score, min inter-car gap) | B6 multi-agent |
| ⚡ **Multi-seed runs with mean±std bands** | A1–A5, C10 — rigorous comparison, not single noisy curves |
| 🚀 **DQN/loss panels** (TD-loss, mean Q, replay stats) once function approximation lands | A1, A2 |
| 🚀 **Episode replay** — scrubbable top-down animation of the cars per episode | C, D — see behavior, not just curves |
| ⚡ **Hyperparameter sweep view** — fan out headless runs, plot the Pareto front | A, C |
| ⚡ **More logged metrics** (lane changes, min gap, time-to-clear, near-misses) | C10 benchmark scorecard |
| ⚡ **Benchmark scorecard tab** (per-scenario results) | C10 |
| ⚡ CSV/PNG export · persist selected-runs/zoom in `localStorage` · report generator | all — sharing results |

---

## If you resume: top 3 for impact/effort
1. **Gymnasium wrapper + DQN** — turns the harness into real, generalizing RL. → dashboard: loss/Q panels + multi-seed bands.
2. **Multi-agent extension** — delivers the actual collaborative thesis. → dashboard: per-agent curves + cooperation metrics.
3. **Reward shaping + de-hardcoded scenarios** — makes learning meaningful and general. → dashboard: benchmark scorecard + episode replay.

> In the v1.0.0 line these land on `caatc/` (the `f1tenth_gym` core): items 1–2 are largely realized by
> the M1 `ClearanceEnv` (a Gymnasium wrapper with a centralized multi-agent action + cooperative
> reward), with training to follow in M2. The dashboard visualizes any run written to
> `saved_variables/runs/`, so most dashboard items are additive. (The legacy headless harness that
> prototyped this on the ROS 1 line lives at tag `v0.3.0`, `tools/rl_harness/train.py`.)
