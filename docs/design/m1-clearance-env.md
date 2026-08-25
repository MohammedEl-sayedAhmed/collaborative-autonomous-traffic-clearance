# M1 — ClearanceEnv design

The v1.0.0 milestone that turns the bare `f1tenth_gym` base (M0) into our actual problem: several
cars cooperating to clear a path for an **emergency vehicle (EV)**. Built as a **Gymnasium wrapper**
over `f1tenth-v0` (no fork). Status: **accepted** (2026-08-26) — both forks confirmed (see below);
implementation in progress. Recorded as [ADR 0007](../adr/0007-m1-clearance-env-design.md).

## Approach — ACC-governed multi-lane move-aside corridor
- **Composition wrapper** `ClearanceEnv(gymnasium.Env)` over
  `gym.make("f1tenth_gym:f1tenth-v0", config={num_agents: K+1, observation_config: {type: kinematic_state}, map: <Track>})`,
  headless (`SDL_VIDEODRIVER=dummy`).
- `agent_0` = the **scripted EV** (kept at the base env's `ego_idx=0` so native ego-collision
  termination applies). `agents 1..K` = the **cooperators** we train (K=3 default).
- A **virtual 3-lane road** via `Track.from_refline` (tiny sinusoid centerline to avoid degenerate
  curvature; `from_refline` has no walls). Lane width ≈ 0.9 m (≈3 car-widths), centers `d ∈ {−0.9, 0, +0.9}`;
  EV lane = center. The wrapper owns a **frenet frame** (`project(x,y)→(s,d)`) for progress, lane index,
  goal, the blocked flag, and an EV off-road clamp (needed since there are no walls).

### The headroom mechanism (the load-bearing idea)
Blocking is enforced by an **adaptive-cruise (ACC) EV law on wide lanes**, *not* a fragile narrow
corridor. Cooperators cruise slow (`V_COOP ≈ 2 m/s`); the EV wants `EV_MAX ≈ 8 m/s`. Any car left in
the EV's lane ahead **clamps the EV to convoy speed** (graded slowness — never a crash, so no
false-failure); a timely move-aside lets the EV **sprint**. A ~4× speed ratio → large, tunable, robust
headroom decoupled from cm-precise low-speed steering.

Cooperators start **staggered** in the EV lane (`s = s0 + j·Δ`), so the EV must pass each in sequence →
**return is monotone in how many yield in time** → a dense gradient that **defeats the saturation trap
by construction** (the exact failure that hid all algorithm differences in the legacy toy harness).

## Interfaces
- **Action (per cooperator):** `Discrete(5)` = {STAY, MERGE_LEFT, MERGE_RIGHT, SPEED_UP, SLOW_DOWN}
  (mirrors the 2020 project's 5-action set; DQN-ready), translated by an internal low-level lane-keeper.
  Joint space `MultiDiscrete([5]*K)`.
- **V2V observation (per cooperator):** SELF (lateral `d`, lane one-hot, speed, heading err, `delta`) +
  **EV broadcast** range-gated to ≈25 m with an `ev_active` flag: relative `Δs`, `Δd`, EV speed,
  **time-to-arrival**, and **intended lane** (intention sharing, ROADMAP #7) + "EV in my corridor &
  behind" flag; + M=2 nearest neighbors' relative state and per-side `left_clear`/`right_clear`
  (ROADMAP #4 — what makes the HARD merge learnable). Default = centralized **joint Box** (immediate
  SB3); optional PettingZoo-parallel Dict + global state in `info` for CTDE in M2.
- **Cooperative reward (shared):** EV progress + EV acceleration term (thesis fidelity) − blocking
  penalty − anti-oscillation (per lane change) − collision penalty.
- **Termination:** success = EV reaches `s_goal`; failure = any car-car collision (from
  `env.unwrapped.collisions`, recomputed by the wrapper) or timeout (`t ≥ T_max`).

## Presets
- **EASY** — side lanes empty; any merge direction is collision-free → a guaranteed performance floor,
  timing-only. **First trainable milestone.**
- **HARD** — side-lane occupants force reading V2V occupancy and choosing the *free* lane; wrong lane =
  collision. Direct port of the legacy "blocker" scenario (which showed random 2% vs learned 100%).

## Baselines
- `naive_hold` — all cooperators STAY (ignore the EV): the reference floor.
- `random_policy` — uniform `Discrete(5)`.
- `ideal_cooperator` — scripted oracle that vacates the EV lane to the correct free side just before
  the EV arrives: the ceiling (used only to prove headroom, never trained).

## Verification — prove headroom BEFORE training
1. **Static math:** naive ⇒ EV clamped to `V_COOP` ⇒ `t_clear ≈ L/2 ≈ 22 s`; ideal ⇒ EV at `EV_MAX`
   ⇒ `t_clear ≈ L/8 ≈ 6 s`; ratio ≈ 4×, fixed by speeds (RNG-independent). Set `T_max` *between* them so
   naive **truncates** (categorical failure) while ideal **succeeds**.
2. **Empirical gate** (`caatc/clearance_smoke.py`, headless, seconds, ≥20 seeds on EASY): assert
   naive success ≈ 0% (or truncates), ideal success = 100%, `mean(t_clear_naive)/mean(t_clear_ideal) ≥ 3`,
   and on HARD `random` has substantial collisions while `ideal` has none. **Exits non-zero if the gap
   is absent — training is never spent on a secretly-saturated scenario.**
3. **Monotonicity check:** a baseline that yields only the first `m` of `K` cooperators shows `t_clear`
   decreasing smoothly as `m: 0→K` — proving dense partial credit (random ≪ optimal, and the optimum is
   reachable).
4. **Dashboard:** emit naive/random/ideal (and later trained) runs as JSONL to `saved_variables/runs/`
   so the existing scorecard shows the band the learner must climb (mean `t_clear`, EV mean speed,
   success %, collision %, lane changes).

## Implementation plan (ordered)
1. `caatc/scenario.py` — `ScenarioConfig` dataclass (geometry/speed/reward/preset knobs) + `EASY_PRESET`
   / `HARD_PRESET`; `build_track(cfg) -> Track` (tiny-sinusoid centerline).
2. `caatc/frenet.py` — `CenterlineFrame` (polyline + cumulative arclength): `project(x,y)->(s,d)`,
   `lane_of(d)`, `lane_center_d(lane)`, `tangent(s)`. Unit test: projection round-trips on the sinusoid.
3. `caatc/controllers.py` — `ev_control(...) -> [steer, speed]` (lane-keep + ACC gate + lateral clamp);
   `coop_lowlevel(target_lane, target_speed, ...) -> [steer, speed]` (P/pure-pursuit lane-keeper). Pure,
   unit-testable.
4. `caatc/clearance_env.py` — `class ClearanceEnv(gymnasium.Env)`: `__init__(cfg)` builds the track +
   inner env + frame; defines `observation_space` / `action_space` + per-cooperator `(target_lane,
   target_speed)` state.
5. `reset(seed, options)` — deterministic poses `(K+1, 3)`: EV at `s=0` center lane; K cooperators
   staggered at `s0+j·Δ` (+ side-lane occupants on HARD); assert non-overlap; `inner.reset(options={'poses': poses})`.
6. `step(action)` — decode joint action → update each cooperator `(target_lane, target_speed)`;
   `coop_lowlevel` → rows 1..K, `ev_control` → row 0; stack `(K+1, 2)`; `inner.step` once; read
   `env.unwrapped.collisions` + poses; frenet-project; compute shared reward; own `terminated` (goal OR
   collision) / `truncated` (`t ≥ T_max`); build V2V-augmented per-cooperator obs (drop `agent_0`).
7. `_build_obs()` — SELF + EV-broadcast (range-gated, time-to-arrival, intended lane) + M-nearest
   neighbors + `left_clear`/`right_clear`; normalize; joint concat or Dict per obs mode.
8. `caatc/baselines.py` — `naive_hold`, `random_policy`, `ideal_cooperator` (each returns a joint action).
9. `caatc/clearance_eval.py` — `run_episode` / `run_batch`; write JSONL to `saved_variables/runs/<label>/`
   (dashboard format); CLI `--policy {naive,random,ideal} --preset {easy,hard} --episodes --seed`.
10. `caatc/clearance_smoke.py` — the pre-training headroom gate (asserts the naive-vs-ideal gap; exits
    non-zero if absent). A `caatc-clearance-smoke` script entry + a `./run.sh` command.
11. `caatc/tests/` — `test_frenet.py`, `test_controllers.py`, `test_termination.py`, `test_headroom.py`.
    Register `caatc/clearance-v0`.

## Open fork — CONFIRMED (2026-08-26)
Both forks were confirmed by the owner; implementation proceeds on these choices:
1. **Blocking mechanism → ACC on wide lanes.** (Not a narrow physical corridor.) ACC decouples
   guaranteed headroom from fragile low-speed steering and keeps the EV crash-free; the corridor route
   would have reworked the EV controller + action space (continuous offset).
2. **First trainable preset → EASY first.** A guaranteed floor, timing-only. (Not straight to HARD's
   occupancy reasoning + wrong-lane collisions — HARD follows once EASY learns.)

*(Design produced by a 4-proposal design workflow — angles: racetrack-overtake, virtual multi-lane road,
minimal-headroom, thesis-faithful V2V — then synthesized.)*
