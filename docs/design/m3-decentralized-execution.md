# M3: each car decides on its own

M3 removes the last central piece in the loop: the **controller**. M2 proved the scenario can be
learned: one PPO network that reads all K cooperators and outputs all their actions matches the
hand-written ideal on both presets. M3 keeps that result and takes the shared view away. Each
cooperating car decides from **its own 26 numbers** (what it senses itself, plus the V2V messages it
receives), with one network run K times, and no shared state at run time. Training may still see
everything (this is called CTDE, "centralised training, decentralised execution"); running the cars
may not. Status: **accepted** on 2026-09-03 and **built**. Recorded as
[ADR 0009](../adr/0009-decentralized-execution-ippo.md).

## The approach: one shared per-car policy, run K times

- **Parameter-shared IPPO on stable-baselines3.** IPPO means "independent PPO learners"; parameter
  sharing means they all use the same network. One `PPO("MlpPolicy")` with
  `observation_space = Box(-10, 10, (26,))` and `action_space = Discrete(5)`. All K cooperators feed
  their experience into one buffer, and share one actor and one critic. There is no "which car am I"
  input: the cars can already be told apart by what they see (a different distance to the EV, a
  different lane, different `left_clear` / `right_clear`).
- **No new run-time dependencies.** The same `caatc-train` image, the same learner class, one fewer
  output head than M2. No PettingZoo, no SuperSuit, no multi-agent library, no hand-written trainer.
- **The plumbing that splits K cars into K streams is ours** (about 110 lines). `AgentSplitVecEnv`
  wraps **M2's own `train.build_vec_env` unchanged** and presents `N` joint scenario environments as
  `N*K` single-car streams. The physics, the seeding, `SubprocVecEnv` and `Monitor` are the exact code
  M2 runs, which is the strongest guarantee we have that the M2 reference is untouched.
- **Training looks like deployment.** The other cars in every rollout *are* copies of the same policy,
  so all K experiences per physics step are valid training data. One physics step gives K samples, and
  the policy we train is the policy we deploy.
- **The deployed object is a `policy(env) -> joint action` function** that does one batched forward
  pass over a `(K, 26)` array: K independent decisions. It drops into `clearance_eval.run_episode`
  unchanged, so learned-per-car, learned-central (M2), naive, random and ideal are all measured by the
  same code, as ADR 0008 requires.

### The key idea

The central observation is already the K per-car observations **glued together**, and each per-car
observation has `12 + 2*num_lanes + 4*num_neighbors = 26` numbers **whatever K is** (it holds the
M=2 nearest neighbours, not K−1). So splitting per car is a **reshape, not a rewrite**:
`(K*26,) → (K, 26)` with one `Discrete(5)` head. The physics, the ACC law, the road frame, the reward
and the end-of-episode rules do not move. That is what makes M3 cheap, and also what makes it
*checkable*: the M2 vs M3 comparison is about **who sees what**, not about different features, and a
test that demands identical numbers enforces exactly that.

Two consequences, stated up front:

1. **M2 already sits at the ideal** (100% success, 0 collisions, `t_clear` 6.00 s against the ideal's
   6.13 s), so M3 **cannot claim to do better**. It claims *the same result with less information*.
   The evidence is **structural** (checks and tests that fail if any shared state is read) plus two
   columns M2 cannot fill at all: **a different number of cars** (the same weights with K ∈ {2, 4})
   and **lost V2V messages per car**. Anything else would be a number dressed up to flatter the method.
2. Because the per-car action is `Discrete(5)`, the reason ADR 0008 rejected DQN goes away. M3 makes
   off-policy learners possible as a side effect (not in scope).

### What M3 does not change

`frenet.py`, `controllers.py`, the 10-tick physics loop, the meaning of the actions, the shared reward,
the end-of-episode rules, `clearance_eval.run_episode` / `run_batch` / `summarize` / `write_run`, the
JSONL format, the dashboard (zero code changes), the three hand-written baselines, `render2d.py`, the
Docker images, and `train.py` (apart from one added keyword). The M2 models still load and still
reproduce their numbers.

## Interfaces

**`caatc/obs_spec.py`** (new): a named layout for the 26 numbers, so a per-car policy, the local
hand-written policy and the tests all agree on what each number means, instead of using magic offsets
(a test used to assert `obs[7] == 1.0`):

```python
def feature_count(cfg) -> int              # 12 + 2*num_lanes + 4*num_neighbors == 26
def obs_layout(cfg) -> Dict[str, slice]    # self_d, lane, v, heading_err, delta, ev_active, ev_ds,
                                           # ev_dd, ev_v, ev_tta, ev_lane, ev_in_my_lane_behind,
                                           # neigh, left_clear, right_clear
@dataclass(frozen=True)
class LocalView: ...                       # decoded, typed, readable
def decode(obs, cfg) -> LocalView
```

**`caatc/clearance_env.py`** (additions only):

```python
@property
def obs_features(self) -> int
def per_agent_obs(self, j, cars=None) -> np.ndarray        # (F,) float32, clipped
def per_agent_obs_all(self, cars=None) -> np.ndarray       # (K, F) float32
def _build_obs(self, obs, cars=None)                       # == per_agent_obs_all(...).reshape(-1)
```

The two are equal by construction (`np.clip` works element by element) and a test,
`test_per_agent_obs_matches_joint_slice`, enforces it. Plus `ScenarioConfig.neighbor_range:
Optional[float] = None` (defaults to `v2v_range`) to **range-limit the neighbour slots**, which were the
one part of the "local" observation that was not limited at all (decision 2 below), and four added
`info` keys the dashboard ignores: `reward_terms`, `collision_agents`, `lane_changes_per_agent`,
`premature_merge`.

**`caatc/decentralized.py`** (new): deployment and enforcement:

```python
class SharedPolicySquad:   # the per-car claim in one class: model.predict on a (K, F) batch
class LocalSquad:          # the hand-written LOCAL policy deployed to all K
class MixedSquad:          # one seat replaced by a car that does not cooperate
class LocalOnlyView:       # a wrapper where only cfg / per_agent_obs* / obs_features survive;
                           # every other attribute raises AttributeError
```

**`caatc/vec_agents.py`** (`AgentSplitVecEnv`), **`caatc/train_dec.py`** (the IPPO entry point),
**`caatc/dec_smoke.py`** (the plumbing + locality check), **`caatc/proc_fleet.py`** (K real
operating-system processes, one per car: the end-to-end proof that nothing shared is needed).

## Checks: the same habit as M1, applied to M3's claim

M1's check proved **headroom** before spending compute. M3's unproven claim is different: *that the
existing 26 numbers are enough without shared state, and that they are truly local*. So the checks
follow that line. Both new checks exit with an error on failure.

1. **`./run.sh clearance-smoke` stays the precondition**, with the same output as before, and gains
   `--num-cooperators K` so every K that M3 *claims* is checked before it is *measured*.
2. **The "is the local view enough" check (`--m3`, seconds, no learning):** `LocalIdealCooperator`
   deployed to all K through the untouched `run_episode` on 20 or more seeds. EASY and HARD success
   ≥ 0.95, HARD collisions 0, `mean(t_clear)` within 10% of the all-seeing ideal. **If a hand-written
   local policy matches the all-seeing ideal, the claim is proven before a single training step.** If
   it does not, M3 is the wrong milestone and the right one is a richer observation (sharing
   intentions, roadmap item 7).
3. **The plumbing + locality check (`./run.sh dec-smoke`):** the observations are identical to M2's;
   the end-of-episode observation is split correctly; the bookkeeping is right (exactly one JSONL
   record per scenario episode, returns on the team scale); the process fleet matches; and
   **`check_locality`**, written so it *can* fail: move a non-EV car from one out-of-range position to
   another (40 m to 45 m) and demand that `per_agent_obs(j)` does not change by a bit. **This failed at
   first**, which is what decision 2 below is about.
4. **The evaluation code is untouched.** Every policy stays a `policy(env) -> action` function, and
   reported rewards always come from `ClearanceEnv.step` itself.

## Success criteria

- **Main (the same result with less information):** on EASY and HARD, ≥ 95% success, 0 collisions, and
  `mean(t_clear)` within 5% of M2's central policy on **shared seeds**, while every locality check
  passes. A real failure would be a *structural* gap (for example collisions appearing because the
  cars cannot see each other's intentions), not a tuning gap.
- **Columns M2 cannot fill:** the same weights deployed at K ∈ {2, 4} (each checked first), and
  graceful degradation when each car's V2V message is lost with probability p ∈ {0, 0.1, 0.3}.
- **Structural:** `test_executor_touches_no_global_state` (the policy runs against `LocalOnlyView`),
  and `proc_fleet` reproducing the in-process numbers within tolerance.

## Results (measured)

**EASY, paired on 50 shared seeds** (every policy sees exactly the same episodes):

| policy | success | collisions | mean `t_clear` | EV speed | return | lane changes |
|--------|--------:|-----------:|---------------:|---------:|-------:|-------------:|
| `naive` (all STAY) | 0% | 0% | — | 2.29 m/s | 31.3 | 0.0 |
| `speedup` (never moves aside) | **100%** | 0% | 10.61 s | 4.18 m/s | 97.6 | 0.0 |
| `ideal` (all-seeing) | 100% | 0% | 6.13 s | 7.30 m/s | 102.2 | 3.0 |
| **M2 central** (joint action) | 100% | 0% | **6.00 s** | 7.43 m/s | 102.29 | 3.0 |
| **M3 per car** (own observation only) | 100% | 0% | 6.58 s | 6.87 m/s | 101.54 | 2.3 |

**HARD, 20 seeds:** the per-car policy reaches **exactly** the central result: 100% success, 0
collisions, `t_clear` 6.00 s, 7.4308 m/s, return 102.264, 3.0 lane changes. The numbers are identical
to the bit because both policies reach the *free-flow optimum*: no car ever blocks the EV, so the EV
drives the same unblocked path. Their action sequences differ (checked), and the networks have
different shapes (19,472 parameters on `(78,)` / `MultiDiscrete` against 12,166 on `(26,)` /
`Discrete`). The equality is in the outcome, not in the mechanism.

**Column 1: a different number of cars** (the *same* weights, at a K they never trained on; M2's joint
policy cannot do this at all, its action space is fixed at K=3):

| K | success | collisions | `t_clear` | lane changes |
|---|--------:|-----------:|----------:|-------------:|
| 2 | 100% | 0% | 10.62 s | **0.0** |
| 3 (trained) | 100% | 0% | 6.44 s | 2.4 |
| 4 | 100% | 0% | 6.48 s | 2.2 |

K=4 transfers cleanly. **K=2 does not**: with 0.0 lane changes it has fallen back to driving as a fast
convoy (see below). It "succeeds" without cooperating, and that is reported as a limitation, not a win.

**Column 2: lost V2V messages** (each car's broadcast lost with probability p, K=3):

| p | 0.0 | 0.1 | 0.3 | 0.5 |
|---|----:|----:|----:|----:|
| success | 100% | 100% | 100% | 100% |
| `t_clear` | 6.44 s | 6.44 s | 6.56 s | 6.80 s |

It degrades gracefully all the way to losing half of all messages. A central controller cannot even be
asked this question, because its input is the whole joint vector.

**STRICT, 20 seeds.** The preset added in
[ADR 0010](../adr/0010-strict-preset-removes-the-convoying-substitution.md), where a car is speed-capped
while still in the EV's lane, so **a fast convoy cannot stand in for moving aside**:

| policy | success | collisions | mean `t_clear` | EV speed | return | lane changes |
|--------|--------:|-----------:|---------------:|---------:|-------:|-------------:|
| `naive` | 0% | 0% | — | 2.29 m/s | 31.4 | 0.0 |
| `speedup` | **0%** | 0% | — | 2.29 m/s | 31.4 | 0.0 |
| `ideal` (all-seeing) | 100% | 0% | 6.12 s | 7.31 m/s | 102.2 | 3.0 |
| **M2 central** | 100% | 0% | 6.13 s | 7.296 m/s | 102.20 | **3.0** |
| **M3 per car** | 100% | 0% | **6.13 s** | **7.296 m/s** | **102.20** | **3.0** |

On STRICT the central and per-car policies give **identical numbers, and both equal the ideal**: 100%
success, zero collisions, and **3.0 yields per episode**. Every car gets out of the way, every episode.
This is the result M3 was built to produce, and it changes how the EASY gap below should be read.

**EASY with the central critic** (`--central-critic`: each car's actor still reads only its own 26
numbers, but the critic reads all K views during training), 20 seeds:

| policy on EASY | success | collisions | mean `t_clear` | EV speed | return | lane changes |
|----------------|--------:|-----------:|---------------:|---------:|-------:|-------------:|
| M3 per car, shared reward (IPPO) | 100% | 0% | 6.44 s | 7.01 m/s | 101.68 | 2.4 |
| **M3 per car, central critic (CTDE)** | 100% | 0% | **6.12 s** | **7.32 m/s** | **102.08** | 2.0 |
| M2 central (reference) | 100% | 0% | 6.00 s | 7.43 m/s | 102.29 | 3.0 |

**+1.9% against M2**, inside the 5% target, against +7.3% for the shared-reward learner on the same 20
seeds.

> **Re-measured on 2026-09-04.** The first CTDE run (on Python 3.11) measured 6.02 s / 7.42 m/s /
> 102.26 with 3.0 yields. That model file could not be loaded on Python 3.12: I had built the policy
> class inside a function, so the saved file stored the class itself instead of a reference to it, and
> such a file only loads under the exact Python that wrote it (it crashed the process instead of giving
> an error). The class is now defined at module level and the model was **retrained on Python 3.12**.
> The numbers above are that run. The conclusion is the same, a central critic closes most of the gap,
> but the margin is +1.9%, not +0.33%.
>
> One detail worth keeping: this run clears **faster** than the shared-reward learner while moving
> aside **fewer** times (2.0 against 2.4). So what matters is not "move aside three times" but "move
> aside early enough". A car that moves before the EV arrives is worth more than one that moves late.

### Verdict against the success criteria, and what the gap is

- **STRICT: met exactly.** Per car equals central equals the ideal, with all 3.0 yields. On the preset
  where success *means* cooperation, deciding per car costs nothing.
- **HARD: met.** The per-car policy equals the central one exactly.
- **EASY: missed with plain IPPO.** `t_clear` is **+9.7%** against M2, against a 5% target. Success and
  collisions are identical; the difference is 2.3 yields per episode instead of 3.0.

**Two separate fixes, each enough on its own, which is what settles the explanation.** The EASY gap had
two causes, and removing *either* closes it:

1. **Remove the loophole** (STRICT): the same learner, same settings, same training budget yields
   3.0 of 3.0 and matches the ideal. So the gap was partly the policy *using* the fast-convoy loophole.
2. **Give better credit** (the central critic on EASY): with the loophole still open, a central critic
   alone brings the policy to within +1.9% of M2 (from +7.3% on the same seeds). So the gap was also a
   real shared-reward credit problem: with one reward for the whole team, a car cannot easily tell how
   much of the outcome was its own doing.

That both work is stronger evidence than either alone, and it confirms the per-car measurement below
(30/30, 25/30, 18/30) as a real credit-assignment pattern and not only a preference for the loophole:
the car furthest from the EV gains the most from letting the others do the work *and* has the most
delayed contribution, and each fix addresses one of those.

*(For the record: after the STRICT result I expected the central critic **not** to help much, since
STRICT seemed to explain the gap by itself. The measurement showed otherwise: CTDE closes it with the
loophole still present. The explanation above is the corrected one.)*

**So the criterion is now met on all three presets:** STRICT and HARD with plain IPPO, EASY with the
central critic. That option is therefore *adopted*, not merely available: `--central-critic` is the
recommended setting on presets that still allow the loophole.

The cause was diagnosed rather than tuned around. Per-car behaviour over 30 episodes: car 1 (nearest
the EV) moves aside **30/30**, car 2 **25/30**, car 3 **18/30**. It falls with distance from the EV,
which is the signature of **shared-reward credit assignment**, not of uniform under-training: the
further ahead a car is, the longer the delay between its move and the EV's payoff, and the more its
contribution is hidden by the others' moves.

Raising γ (how much the learner values future reward) was tried and made things **worse**, in an
informative way: at γ=0.999 the policy stopped moving aside altogether (0.0 lane changes, `t_clear`
10.6 s), because the end-of-episode success bonus is then worth the same whenever it arrives. That
exposed a property of the **scenario**, not of the learner:

> **The scenario has a loophole.** Since the EV's adaptive-cruise law follows whatever is in front of
> it, cooperators that merely *speed up* let it through without anyone moving aside: on both presets
> `speedup` reaches **100% success** and about 95% of the ideal's return. **So the success rate cannot
> tell cooperation from a fast convoy.** Only the clearance time can. The M1 check never showed this
> because it compared all-STAY against the ideal and nothing in between.

That baseline is now part of `baselines.py` and of the headroom check (`cooperating beats convoying`,
on speed and on return), so the loophole stays visible and any future scenario change is checked
against it. Closing the loophole (a shorter time limit, a heavier blocking penalty, or a speed cap
while in the EV's lane) changes the scenario and would invalidate M2's published numbers, so it was
done as a *new* preset, STRICT, rather than as a quiet edit here.

## Implementation plan (in order, with decision points; all done)

1. **Step 0, the claim** (half a day, no compute): `obs_spec.py`; `per_agent_obs*`, the `_build_obs`
   refactor and the identical-numbers test; the `neighbor_range` limit and its "changes nothing" test;
   `LocalIdealCooperator`, `LocalSquad`, `LocalOnlyView`; the `--m3` check.
   **Decision point 1:** the local hand-written policy matches the all-seeing one on EASY *and* HARD, or
   stop and re-scope M3 as a richer observation.
2. **Step 1, the plumbing** (half a day, about 5 min compute): `vec_agents.py`, `train_dec.py`,
   `dec_smoke.py`, `proc_fleet.py`, and a 30k-step EASY training smoke test.
   **Decision point 2:** `dec-smoke` passes all five checks.
3. **Step 2, EASY** (about 1 h): 300k scenario steps × 3 seeds; paired evaluation against M2 on shared
   seeds.
4. **Step 3, HARD and the extra columns** (about 1.5 h): 300k to 600k × 3 seeds; the K sweep; the
   message-loss sweep; the mixed-squad column.
5. **Step 4, the record:** this doc, ADR 0009, a note in `docs/DASHBOARD.md`, `run.sh` and
   `pyproject` entries.
6. **Kept in reserve at first, then built:** the central-critic option, `obs = concat(own 26, all
   K*26)` with a custom `ActorCriticPolicy` whose **actor reads only the first 26 numbers** while the
   critic reads everything. MAPPO's variance reduction without MAPPO, still deployable per car (about
   80 lines). Adopted once the EASY measurement showed the need.

## The three open questions, decided on 2026-09-03

1. **The learner: parameter-shared IPPO.** It trains under the condition it is deployed in, uses all
   K experiences per physics step instead of one, and needs no partner pool, stage schedule or
   cross-play table (about 300 lines and 49 run folders less), keeping SB3's PPO as the maintained
   learner. The "other cars keep changing" problem is real but mild here: with `coop_gap = 10 m` against
   `clear_window = 6 m`, two cars that have merged never sit in each other's clear window. If IPPO
   stalled, the next step was the central-critic policy above, not the partner ladder.
2. **Range-limit the neighbour slots now: yes**, `neighbor_range = None → v2v_range`. Without it, "no
   car sees the whole picture" is not true, and the locality check cannot be written in a form that can
   fail. Measured on master at `f4534bb` (ideal-policy rollouts, 4 seeds): the largest `|Δs|` ever
   carried in a neighbour slot is **20.84 m on EASY / 10.56 m on HARD**, and **0 of 1,476 slots** exceed
   the 25 m range on any default setting. So the fix **changes nothing on any setting the M2
   comparison uses**, and a test proves the numbers are identical. But at `coop_gap = 13 m` (+30%,
   exactly the sweep this milestone planned) **492 of 1,476 slots carry a car beyond range, up to
   26.86 m**. The leak is real, and it would have bitten exactly where the new claims live, invisibly.
3. **Build the PettingZoo mode in M3: yes**, keeping ADR 0007's promise now rather than later. It is
   `caatc/pz_env.py`, a `ParallelEnv` over `per_agent_obs_all()`, with `pettingzoo` as an optional,
   test-only dependency and its own **conformance tests** (`pettingzoo.test.parallel_api_test`), so
   it is exercised code rather than a pin with no caller. Training still goes through
   `AgentSplitVecEnv`; the adapter is the hook for outside multi-agent libraries (QMIX, MADDPG,
   roadmap item 6).
