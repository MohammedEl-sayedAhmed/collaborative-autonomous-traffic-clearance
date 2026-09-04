# M3 — Decentralized execution design

The v1.0.0 milestone that removes the last centralized thing in the loop: the **controller**. M2 proved
the scenario is learnable — one PPO network reading all K cooperators and emitting all their actions
matches the scripted oracle on both presets. M3 keeps that result and takes the joint view away: each
cooperating car decides from **its own 26-feature observation** (its own sensing plus the V2V broadcast
it receives), one network evaluated K times, no joint state at execution. Centralized *training* stays
(CTDE); centralized *execution* goes. Status: **accepted** (2026-09-03) — all three forks confirmed (see below).
Recorded as [ADR 0009](../adr/0009-decentralized-execution-ippo.md).

## Approach — one shared per-agent policy, K independent forward passes

- **Parameter-shared IPPO on stable-baselines3.** One `PPO("MlpPolicy")` with
  `observation_space = Box(-10, 10, (26,))` and `action_space = Discrete(5)`. All K cooperators are
  independent transition streams into one rollout buffer sharing one actor and critic. No agent-id
  feature: the seats are already distinguishable from the observation itself (different EV Δs, lane
  one-hot, `left_clear`/`right_clear`).
- **Zero new runtime dependencies** — the same `caatc-train` image, the same learner class, one fewer
  action head than M2. No PettingZoo, no SuperSuit, no MARL library, no hand-rolled trainer.
- **The K→streams transport is ours** (~110 lines): `AgentSplitVecEnv` wraps **M2's own
  `train.build_vec_env` verbatim** and presents `N` joint scenario envs as `N*K` single-agent streams.
  Physics, seeding, `SubprocVecEnv` and `Monitor` stay the code M2 runs — the strongest available
  guarantee that the comparison reference is intact.
- **Deployment is the training condition.** The partners in every rollout *are* copies of the same
  stochastic policy, so their transitions are on-policy and all K are harvested: one physics step
  yields K gradient samples, and the fixed point optimized is the one deployed.
- **The deployed object is a `policy(env) -> joint action` callable** doing one batched forward pass
  over `(K, 26)` — K independent decisions. It drops into `clearance_eval.run_episode` unchanged, so
  learned-decentralized, learned-centralized (M2), naive, random and ideal are all measured by
  identical code, as ADR 0008 requires.

### The load-bearing idea

`_build_obs` is already **agent-major** — literally `concat(_per_coop_obs(0..K-1))` — and the per-agent
width `_F = 12 + 2*num_lanes + 4*num_neighbors = 26` is **independent of K** (M=2 neighbour slots, not
K−1). So decentralization is a **reshape, not a rewrite**: `(K*26,) → (K, 26)` with one `Discrete(5)`
head. Nothing about the physics, the ACC law, the frenet frame, the reward or the termination semantics
moves. That is what makes M3 cheap, and also what makes it *checkable*: the M2↔M3 comparison is about
**information access**, not features, and a bit-identity test enforces exactly that.

Two consequences, stated up front:

1. **M2 already sits at the oracle ceiling** (100% success, 0 collisions, `t_clear` 6.00 s vs the
   oracle's 6.13 s), so M3 **cannot be an improvement claim**. It is an *equality-under-information-
   restriction* claim, whose evidence is **structural** (gates and tests that fail if any global state
   is read) plus two capability columns M2 structurally cannot fill: **K-transfer** (the same weights
   deployed at K ∈ {2, 4}) and **per-agent V2V dropout**. Anything else would be a metric dressed up to
   flatter the method.
2. Because the per-agent action is `Discrete(5)`, the reason ADR 0008 rejected DQN disappears — M3
   incidentally unlocks off-policy learners (not in scope).

### What M3 does not change

`frenet.py`, `controllers.py`, the 10-substep physics loop, the action semantics, the shared reward,
the termination rules, `clearance_eval.run_episode`/`run_batch`/`summarize`/`write_run`, the JSONL
schema, the dashboard (zero code changes), the three scripted baselines, `render2d.py`, the Docker
images, and `train.py` (bar one additive keyword). The M2 models still load and still reproduce their
numbers.

## Interfaces

**`caatc/obs_spec.py`** (new) — a named wire format, so a decentralized car, the local oracle and the
tests all agree on the layout instead of using magic offsets (`test_termination.py` currently asserts
`obs[7] == 1.0`):

```python
def feature_count(cfg) -> int              # 12 + 2*num_lanes + 4*num_neighbors == 26
def obs_layout(cfg) -> Dict[str, slice]    # self_d, lane, v, heading_err, delta, ev_active, ev_ds,
                                           # ev_dd, ev_v, ev_tta, ev_lane, ev_in_my_lane_behind,
                                           # neigh, left_clear, right_clear
@dataclass(frozen=True)
class LocalView: ...                       # decoded, typed, readable
def decode(obs, cfg) -> LocalView
```

**`caatc/clearance_env.py`** (additive only):

```python
@property
def obs_features(self) -> int
def per_agent_obs(self, j, cars=None) -> np.ndarray        # (F,) float32, clipped
def per_agent_obs_all(self, cars=None) -> np.ndarray       # (K, F) float32
def _build_obs(self, obs, cars=None)                       # == per_agent_obs_all(...).reshape(-1)
```

Equivalence holds by construction (`np.clip` is elementwise) and is enforced by
`test_per_agent_obs_matches_joint_slice`. Plus `ScenarioConfig.neighbor_range: Optional[float] = None`
(→ `v2v_range`) to **range-gate the neighbour block** — today the one part of the "local" observation
that is not gated at all (fork 2) — and four additive `info` keys the dashboard ignores:
`reward_terms`, `collision_agents`, `lane_changes_per_agent`, `premature_merge`.

**`caatc/decentralized.py`** (new) — deployment and enforcement:

```python
class SharedPolicySquad:   # the decentralization claim, in one class: model.predict on a (K, F) batch
class LocalSquad:          # the scripted LOCAL oracle deployed to all K
class MixedSquad:          # one seat replaced by a non-compliant peer
class LocalOnlyView:       # a facade where only cfg / per_agent_obs* / obs_features survive;
                           # every other attribute raises AttributeError
```

**`caatc/vec_agents.py`** (`AgentSplitVecEnv`), **`caatc/train_dec.py`** (the IPPO entry point),
**`caatc/dec_smoke.py`** (the interface + locality gate), **`caatc/proc_fleet.py`** (K real OS
processes, one per car — the end-to-end proof that nothing shared is required).

## Verification — the culture preserved, extended to M3's premise

M1's gate proved **headroom** before spending compute. M3's unproven premise is different — *that the
existing 26-vector is sufficient without global state, and that it is genuinely local* — so the gates
extend along that axis. Both new gates exit non-zero on failure.

1. **`./run.sh clearance-smoke` stays the precondition**, unchanged in default output, gaining
   `--num-cooperators K` so every K that M3 *claims* is gated before it is *evaluated*.
2. **The observability gate (`--m3`, seconds, no learning):** `LocalIdealCooperator` deployed to all K
   through the untouched `run_episode` on ≥20 seeds — EASY and HARD success ≥ 0.95, HARD collisions 0,
   `mean(t_clear)` within 10% of the privileged oracle. **If a hand-written local policy matches the
   privileged oracle, the premise is proven before a single gradient step**; if it does not, M3 is the
   wrong milestone and the right one is observation enrichment (intention broadcast, ROADMAP #7).
3. **The interface + locality gate (`./run.sh dec-smoke`):** obs bit-equivalence with M2, terminal-obs
   split, episode accounting (exactly one JSONL record per scenario episode, team-scale returns), the
   process-fleet match, and **`check_locality`** — written so it *can* fail: teleport a non-EV car from
   one out-of-range position to another (40 m → 45 m) and require `per_agent_obs(j)` to be
   bit-identical. **This fails today**, which is what fork 2 is about.
4. **The eval path is untouched** — every policy stays a `policy(env) -> action` callable, and reported
   rewards always come from `ClearanceEnv.step` itself.

## Success criteria

- **Primary (equality under restriction):** on EASY and HARD, ≥ 95% success, 0 collisions, and
  `mean(t_clear)` within 5% of M2's centralized policy on **shared eval seeds** — while every locality
  gate passes. Genuine failure = a *structural* gap (e.g. collisions appear because cars cannot see
  each other's intent), not a tuning gap.
- **Capability columns M2 cannot fill:** the same weights deployed at K ∈ {2, 4} (each gated first),
  and graceful degradation under per-agent V2V dropout p ∈ {0, 0.1, 0.3}.
- **Structural:** `test_executor_touches_no_global_state` (the policy runs against `LocalOnlyView`),
  and `proc_fleet` reproducing in-process metrics within tolerance.


## Results (measured)

**EASY, paired on 50 shared seeds** (every policy sees the identical episodes):

| policy | success | collisions | mean `t_clear` | EV speed | return | lane changes |
|--------|--------:|-----------:|---------------:|---------:|-------:|-------------:|
| `naive` (all STAY) | 0% | 0% | — | 2.29 m/s | 31.3 | 0.0 |
| `speedup` (never yields) | **100%** | 0% | 10.61 s | 4.18 m/s | 97.6 | 0.0 |
| `ideal` (privileged oracle) | 100% | 0% | 6.13 s | 7.30 m/s | 102.2 | 3.0 |
| **M2 centralized** (joint action) | 100% | 0% | **6.00 s** | 7.43 m/s | 102.29 | 3.0 |
| **M3 decentralized** (own obs only) | 100% | 0% | 6.58 s | 6.87 m/s | 101.54 | 2.3 |

**HARD, 20 seeds:** the decentralized policy reaches **exactly** the centralized result — 100%
success, 0 collisions, `t_clear` 6.00 s, 7.4308 m/s, return 102.264, 3.0 lane changes. The metrics are
bit-identical because both policies attain the *free-flow optimum*: no car ever blocks the EV, so the
EV runs the same unblocked trajectory. Their action sequences differ (verified), and the networks are
different shapes (19,472 params on `(78,)`/`MultiDiscrete` vs 12,166 on `(26,)`/`Discrete`) — the
equality is in the outcome, not the mechanism.

**Capability column 1 — K-transfer** (the *same* weights, at a K it never trained on; M2's joint
policy structurally cannot do this, its action space is fixed at K=3):

| K | success | collisions | `t_clear` | lane changes |
|---|--------:|-----------:|----------:|-------------:|
| 2 | 100% | 0% | 10.62 s | **0.0** |
| 3 (trained) | 100% | 0% | 6.44 s | 2.4 |
| 4 | 100% | 0% | 6.48 s | 2.2 |

K=4 transfers cleanly. **K=2 does not**: with 0.0 lane changes it has fallen back on convoying (see
below) — it "succeeds" without cooperating, and that is reported as a limitation, not a win.

**Capability column 2 — per-car V2V dropout** (each car's broadcast lost with probability p, K=3):

| p | 0.0 | 0.1 | 0.3 | 0.5 |
|---|----:|----:|----:|----:|
| success | 100% | 100% | 100% | 100% |
| `t_clear` | 6.44 s | 6.44 s | 6.56 s | 6.80 s |

Degradation is graceful all the way to losing half of all broadcasts — a robustness claim a joint
controller cannot even be asked about, because its observation is the joint vector.

### Verdict against the success criteria, and what the gap is

- **HARD: the criterion is met** — the decentralized policy equals the centralized one exactly.
- **EASY: the criterion is missed** — `t_clear` is **+9.7%** vs M2, against a 5% bar. Success and
  collisions are identical; the difference is 2.3 yields per episode instead of 3.0.

The cause was diagnosed rather than tuned around. Per-seat behaviour over 30 episodes: car 1 (nearest
the EV) yields **30/30**, car 2 **25/30**, car 3 **18/30** — monotone in distance from the EV, which is
the signature of **shared-reward credit assignment**, not of uniform under-training: the further ahead
a car is, the longer the delay between its yield and the EV's payoff, and the more its contribution is
masked by the others' yields.

Raising γ was tried and made it **worse**, informatively: at γ=0.999 the policy stopped yielding
altogether (0.0 lane changes, `t_clear` 10.6 s) because the terminal success bonus is then valued
whenever it arrives. That exposed a property of the **scenario**, not of the learner:

> **The scenario admits a degenerate strategy.** Since the EV's adaptive-cruise law follows whatever is
> in front of it, cooperators that merely *speed up* let it through without anyone yielding: on both
> presets `speedup` reaches **100% success** and ~95% of the oracle's return. **Success rate therefore
> cannot distinguish cooperation from convoying** — only clearance time can. The M1 gate never showed
> this because it compared all-STAY against ideal and nothing in between.

That baseline is now part of `baselines.py` and of the headroom gate (`cooperating beats convoying`, on
speed and on return), so the substitution is permanently visible and a future scenario change is
checked against it. Whether to *remove* the substitution — a tighter `T_max`, a heavier blocking
penalty, or a speed cap while in the EV's lane — is a scenario change that would invalidate M2's
published numbers, so it belongs to M4 and to the owner, not to a quiet edit here.

**The recorded escalation is therefore indicated for EASY**, on demonstrated need rather than on
aesthetics: the reserve central-critic variant (actor reads only its own F features, critic reads the
joint vector) targets exactly this credit-assignment gap while remaining deployable per-agent.

## Implementation plan (ordered, staged with decision gates)

1. **Stage 0 — the premise** (half a day, no compute): `obs_spec.py`; `per_agent_obs*` + the
   `_build_obs` refactor + bit-identity test; the `neighbor_range` gate + its inertness test;
   `LocalIdealCooperator` + `LocalSquad` + `LocalOnlyView`; the `--m3` gate.
   **Decision gate 1:** the local oracle matches the privileged oracle on EASY *and* HARD, or stop and
   re-scope M3 as observation enrichment.
2. **Stage 1 — the transport** (half a day, ~5 min compute): `vec_agents.py`, `train_dec.py`,
   `dec_smoke.py`, `proc_fleet.py`, and a 30k-step EASY smoke train.
   **Decision gate 2:** `dec-smoke` passes all five checks.
3. **Stage 2 — EASY** (~1 h): 300k scenario-steps × 3 seeds; paired eval against M2 on shared seeds.
4. **Stage 3 — HARD + the capability columns** (~1.5 h): 300–600k × 3 seeds; the K-transfer sweep; the
   dropout sweep; the mixed-squad column.
5. **Stage 4 — the record:** this doc, ADR 0009, a `docs/DASHBOARD.md` note, `run.sh`/`pyproject`
   entries.
6. **Held in reserve, not built:** if IPPO plateaus, the zero-dependency CTDE escalation is
   `obs = concat(ego F, joint K*F)` with a custom `ActorCriticPolicy` whose **actor reads only the
   first F features** while the critic reads everything — MAPPO's variance reduction without MAPPO,
   still deployable per-agent (~80 lines). Adopted only on demonstrated need.

## Open forks — CONFIRMED (2026-09-03)

1. **The learner → parameter-shared IPPO** *(confirmed)*. It trains under the condition it deploys in, harvests all K transitions per
   physics step instead of one, and needs no partner pool, stage schedule or cross-play matrix (~300
   lines and ~49 run directories less), keeping SB3's PPO as the maintained learner. Non-stationarity
   is real but mild here: with `coop_gap = 10 m` against `clear_window = 6 m`, two merged cooperators
   never occupy each other's clear window. If IPPO plateaus, the escalation is the reserve
   central-critic policy above, not the ladder.
2. **Range-gate the neighbour block now → yes** *(confirmed)*, `neighbor_range = None → v2v_range`. Without it, "no global state at
   execution" is not true and the locality check cannot be written in falsifiable form. Measured on
   master @ `f4534bb` (oracle rollouts, 4 seeds): the largest `|Δs|` ever carried in a neighbour slot is
   **20.84 m on EASY / 10.56 m on HARD**, and **0 of 1,476 slots** exceed the 25 m range on any default
   configuration — so the fix is **numerically inert on every configuration the M2 comparison uses**,
   and a bit-identity test can prove it. But at `coop_gap = 13 m` (+30%, exactly the sweep this
   milestone plans) **492 of 1,476 slots carry a car beyond range, up to 26.86 m** — the leak is real
   and would bite precisely where the new capability claims live, invisibly.
3. **Build the PettingZoo parallel mode in M3 → yes** *(confirmed; the owner chose to honour ADR
   0007's promise now rather than defer it).* It lands as `caatc/pz_env.py`, a `ParallelEnv` over
   `per_agent_obs_all()`, with `pettingzoo` as an optional/test-only dependency and its own
   **API-conformance tests** (`pettingzoo.test.parallel_api_test`) so it is exercised code rather than
   a pin with no caller. Training still routes through `AgentSplitVecEnv`; the adapter is the seam for
   external MARL baselines (QMIX/MADDPG, ROADMAP #6).

*(Design produced by a 4-proposal judged workflow — angles: minimal per-agent Gym view, shared IPPO
over a split vec env, true CTDE/MAPPO, V2V-realism-first — then synthesized: the interfaces and
verification come from the winning angle, the learner from the runner-up.)*
