# M1: the ClearanceEnv design

M1 is the milestone that turns the bare `f1tenth_gym` simulator (M0) into our real problem: several
cars working together to clear a path for an **emergency vehicle (EV)**. It is a **Gymnasium wrapper**
around `f1tenth_gym` (we did not fork it). Status: **accepted** on 2026-08-26 and **built**. Recorded as
[ADR 0007](../adr/0007-m1-clearance-env-design.md).

> **As built.** This is the design as written before coding. The code follows it, with a few
> differences worth knowing: the inner `F110Env` is built directly (no `gym.make`), so no extra
> wrappers get in the way; one `step()` runs **10 physics ticks** at 100 Hz (so decisions are taken at
> 10 Hz); a third preset, **STRICT**, was added later ([ADR 0010](../adr/0010-strict-preset-removes-the-convoying-substitution.md));
> and the road frame is built from our own open centre line, not from the track's, because
> `Track.from_refline` quietly closes an open line into a loop (see `docs/upstream-candidates.md`).

## The approach: adaptive cruise on a wide, multi-lane road

- A **wrapper** class `ClearanceEnv(gymnasium.Env)` around the f1tenth simulator with `K+1` cars and
  the `kinematic_state` observation, running without a screen (`SDL_VIDEODRIVER=dummy`).
- `agent_0` is the **scripted EV**. It stays at the simulator's `ego_idx=0`, so the simulator's own
  collision handling applies to it. Cars `1..K` are the **cooperators** we train (K=3 by default).
- A **virtual 3-lane road** made with `Track.from_refline` from a gentle sine-wave centre line (a
  perfectly straight line would have zero curvature everywhere, which some of the maths dislikes).
  `from_refline` builds no walls. Lanes are 0.9 m wide (about 3 car widths), with centres at
  `d ∈ {−0.9, 0, +0.9}` m from the centre line. The EV lane is the middle one. The wrapper keeps a
  **road frame**: `project(x, y) → (s, d)`, where `s` is the distance along the road and `d` the offset
  from the centre line. It gives progress, lane index, the goal, the "blocked" flag, and keeps the EV
  on the road (needed, since there are no walls).

### The key idea: where the headroom comes from

Blocking is done with an **adaptive-cruise (ACC) law on the EV**, on wide lanes, not with a narrow
corridor that needs delicate steering. Our cars drive slowly (`V_COOP ≈ 2 m/s`); the EV wants to go
`EV_MAX ≈ 8 m/s`. Any car left in the EV's lane ahead of it **holds the EV down to convoy speed**
(gradually, never a crash, so there are no false failures). A car that moves aside in time lets the EV
**sprint**. The 4× speed ratio gives a large, tunable, robust headroom that does not depend on
centimetre-precise low-speed steering.

The cooperators start **spread out** along the EV lane (`s = s0 + j·Δ`), so the EV has to pass each one
in turn. That makes the **return grow with every car that moves aside in time**: a smooth signal for
the learner, and the opposite of the trap in the 2020 project, where the task was so easy that random
actions looked optimal and no algorithm difference could show.

## Interfaces

- **Action (per cooperator):** `Discrete(5)` = {STAY, MERGE_LEFT, MERGE_RIGHT, SPEED_UP, SLOW_DOWN}.
  This mirrors the 2020 project's five actions. A built-in low-level lane keeper turns the choice into
  steering and speed. The joint action is `MultiDiscrete([5]*K)`.
- **V2V observation (per cooperator):** SELF (offset `d`, lane as one-hot, speed, heading error,
  steering angle `delta`); the **EV broadcast**, only within about 25 m and with an `ev_active` flag:
  relative `Δs`, `Δd`, EV speed, **time until the EV arrives**, the EV's **intended lane** (this is the
  "share intentions" idea, roadmap item 7), and an "EV is in my lane and behind me" flag; the M=2
  nearest neighbours' relative state; and `left_clear` / `right_clear` for the two side lanes (roadmap
  item 4, and what makes the HARD merge learnable). By default the K observations are glued into one
  joint `Box` for a central learner; a PettingZoo mode came later in M3.
- **Shared reward:** EV progress, plus an EV speed term (as in the thesis), minus a blocking penalty,
  minus a penalty per lane change (to stop wobbling), minus a collision penalty, plus a bonus on success.
- **End of episode:** success when the EV reaches `s_goal`; failure on any car-to-car collision (read
  from the simulator's collision flags) or when time runs out (`t ≥ T_max`).

## Presets

- **EASY**: side lanes empty. Any merge direction is safe, so this is a guaranteed floor and only
  timing matters. **The first preset to train on.**
- **HARD**: side-lane occupants force the cars to read which side is free from V2V and pick it. The
  wrong lane means a collision. This is a direct port of the 2020 "blocker" scenario (where random got
  2% and the learned policy 100%).

## Baselines

- `naive`: all cooperators STAY and ignore the EV. The floor.
- `random`: uniform random over the 5 actions.
- `ideal`: a hand-written policy with full knowledge that leaves the EV lane to the free side just
  before the EV arrives. The ceiling. Used only to prove headroom, never trained.

## Checks: prove the headroom BEFORE training

1. **On paper:** with `naive`, the EV is held at `V_COOP`, so `t_clear ≈ L/2 ≈ 22 s`. With `ideal` it
   runs at `EV_MAX`, so `t_clear ≈ L/8 ≈ 6 s`. The ratio is about 4× and fixed by the speeds (no
   randomness involved). `T_max` is set *between* the two, so `naive` **runs out of time** while `ideal`
   **succeeds**.
2. **In the simulator** (`caatc/clearance_smoke.py`, no screen, takes seconds, 20 or more seeds on
   EASY): `naive` success about 0% (or times out), `ideal` success 100%, `mean(t_clear_naive) /
   mean(t_clear_ideal) ≥ 3`, and on HARD `random` has many collisions while `ideal` has none. **It exits
   with an error if the gap is missing, so training is never spent on a secretly easy scenario.**
3. **Smooth partial credit:** a baseline where only the first `m` of the `K` cars move aside shows
   `t_clear` falling steadily as `m` goes from 0 to K. So random is far below optimal, and the optimum
   is reachable.
4. **Dashboard:** write naive / random / ideal (and later trained) runs as JSONL to
   `saved_variables/runs/`, so the dashboard shows the band the learner has to climb (mean `t_clear`, EV
   mean speed, success %, collision %, lane changes).

## Implementation plan (in order; all done)

1. `caatc/scenario.py`: the `ScenarioConfig` dataclass (road, speeds, reward weights, preset) and the
   presets; `build_track(cfg)`.
2. `caatc/frenet.py`: `CenterlineFrame` (a polyline with cumulative distance): `project(x, y) → (s, d)`,
   `lane_of(d)`, `lane_center_d(lane)`, `tangent(s)`. Unit test: projection round-trips on the sine wave.
3. `caatc/controllers.py`: `ev_control(...) → [steer, speed]` (lane keeping + the ACC law);
   `coop_lowlevel(target_lane, target_speed, ...) → [steer, speed]` (a lane keeper). Pure functions,
   easy to test.
4. `caatc/clearance_env.py`: `class ClearanceEnv(gymnasium.Env)`: `__init__(cfg)` builds the track, the
   inner simulator and the road frame; defines the observation and action spaces and each cooperator's
   `(target_lane, target_speed)` state.
5. `reset(seed, options)`: start poses `(K+1, 3)`: the EV at `s=0` in the middle lane; K cooperators
   spread at `s0 + j·Δ` (plus side-lane occupants on HARD); a check that no two cars overlap;
   `inner.reset(options={'poses': poses})`.
6. `step(action)`: decode the joint action into each car's `(target_lane, target_speed)`; build one
   `[steer, speed]` row per car (`coop_lowlevel` for rows 1..K, `ev_control` for row 0); step the
   simulator; read collisions and poses; project into the road frame; compute the shared reward; set
   `terminated` (goal or collision) and `truncated` (`t ≥ T_max`); build the per-car observations.
7. `_build_obs()`: SELF + EV broadcast (range-limited, arrival time, intended lane) + nearest neighbours
   + `left_clear` / `right_clear`; normalised; glued together.
8. `caatc/baselines.py`: `naive`, `random`, `ideal` (each returns a joint action).
9. `caatc/clearance_eval.py`: `run_episode` / `run_batch`; write JSONL to `saved_variables/runs/<label>/`
   (the dashboard format); command line `--policy {naive,random,ideal} --preset {easy,hard} --episodes --seed`.
10. `caatc/clearance_smoke.py`: the headroom check (fails loudly if the gap is missing), plus a
    `./run.sh` command.
11. `caatc/tests/`: `test_frenet.py`, `test_controllers.py`, `test_termination.py`, `test_headroom.py`.

## The two open questions, decided on 2026-08-26

1. **How blocking works: ACC on wide lanes.** Not a narrow corridor. ACC keeps the headroom
   independent of delicate low-speed steering and keeps the EV crash-free. The corridor idea would have
   needed a new EV controller and a continuous action (an offset).
2. **First preset: EASY.** A guaranteed floor where only timing matters. HARD, with its "read which
   side is free" reasoning and wrong-lane collisions, followed once EASY learned.
