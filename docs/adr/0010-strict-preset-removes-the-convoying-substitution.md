# 0010. A STRICT preset that removes the convoying substitution

- **Status:** accepted
- **Date:** 2026-09-04
- **Deciders:** Mohammed El-sayed Ahmed

## Context and problem statement

M3 found that the scenario admits a **degenerate strategy**. The emergency vehicle's adaptive-cruise
law follows whatever is ahead of it, so cooperators that never yield but simply **speed up** let it
through anyway: on EASY and HARD alike, `speedup` reaches **100% success** at ~95% of the oracle's
return, with **zero** lane changes.

That has two consequences. First, **success rate cannot distinguish cooperation from convoying** — the
project's headline claim ("the cars clear a path") is not what success measures. Second, a learner can
find the shortcut: M3's decentralized policy yielded 2.3 times per episode instead of 3.0 and cleared
9.7% slower than the centralized one, and at γ=0.999 it stopped yielding altogether.

The M1 headroom gate never exposed this because it compared all-STAY against the ideal oracle and
nothing in between.

## Considered options

1. **Add a STRICT preset with an in-lane speed cap** — a cooperator's speed is capped while it is still
   in the EV's lane, so you cannot outrun the ambulance in its own lane. EASY and HARD are untouched.
   **Chosen.**
2. **Tighten `T_max` on the existing presets** — rejected as the primary fix: it removes the shortcut
   only by making the clock harsher for everyone, compressing the learner's margin, and it invalidates
   every published M2/M3 number.
3. **Increase the blocking penalty** — rejected: it changes returns but not *success*, so the metric
   that cannot tell cooperation from convoying stays broken.
4. **Change EASY/HARD directly and re-baseline** — rejected for now: it would invalidate the M2-vs-M3
   comparison as documented and require retraining every published policy. Left available if the owner
   later wants a single canonical scenario.
5. **Leave it, keep the baseline visible** — rejected as insufficient on its own, though the `speedup`
   baseline and its two gate checks are kept regardless.

## Decision

Add **STRICT**: EASY's geometry plus `ScenarioConfig.ev_lane_speed_cap` (set to `coop_speed`), which
caps a cooperator's speed **only while it is in the EV's lane**. A car that has yielded is free to
accelerate again, so the cap targets the substitution rather than the dynamics.

Measured over 20 seeds:

| preset | policy | success | `t_clear` | EV speed |
|--------|-------:|--------:|----------:|---------:|
| EASY | `speedup` | 100% | 10.61 s | 4.18 m/s |
| STRICT | `speedup` | **0%** | — | 2.29 m/s (identical to `naive`) |
| STRICT | `ideal` | 100% | 6.12 s | 7.31 m/s (unchanged) |

The headroom gate gains two STRICT checks — *convoying cannot succeed* and *yielding still succeeds* —
so the property is enforced rather than assumed. `--preset strict` is accepted by every command that
takes a preset — `clearance-eval`, `clearance-train`, `clearance-train-dec`, `dec-smoke` and
`clearance-watch`. `clearance-smoke` takes no `--preset`: it gates *all* presets in one run, and its
STRICT block is part of that fixed sequence.

## Consequences

- **Positive:** on STRICT, success *means* cooperation. It immediately paid for itself: the
  decentralized policy trained on STRICT reaches **exactly** the oracle (100% success, 0 collisions,
  `t_clear` 6.13 s, 7.296 m/s, return 102.20, **3.0** yields) and is **bit-identical to the centralized
  policy** — so M3's +9.7% EASY gap was the learner exploiting the shortcut, not a failure of
  decentralization. Future scenario changes are checked against the `speedup` baseline.
- **Negative / trade-offs:** a third preset to maintain and train on; the cap is a modelling
  simplification (a real car *could* accelerate in-lane, it just would not help an ambulance behind
  it); EASY and HARD keep the shortcut, so results on them must always be read with clearance time
  rather than success rate.
- **Neutral:** EASY/HARD numbers, models and the published M2-vs-M3 comparison are untouched by design.
