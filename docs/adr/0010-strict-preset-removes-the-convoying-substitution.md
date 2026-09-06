# 0010. The STRICT preset: speeding up must not count as cooperation

- **Status:** accepted
- **Date:** 2026-09-04
- **Deciders:** Mohammed El-sayed Ahmed

## The problem

M3 showed that the scenario has a **loophole**. The emergency vehicle follows whatever is in front of
it, so our cars can let it through by just **speeding up**, without ever moving aside. On both EASY
and HARD, a `speedup` policy gets **100% success** at about 95% of the ideal's return, with **zero**
lane changes.

That has two consequences. First, **the success rate cannot tell cooperation from a fast convoy**. The
project's headline claim ("the cars clear a path") is not what success measures. Second, a learner can
find the loophole: the M3 per-car policy moved aside 2.3 times per episode instead of 3.0 and cleared
9.7% slower than the central one, and with γ=0.999 it stopped moving aside altogether.

The M1 headroom check never caught this, because it only compared "nobody moves" with the ideal, and
nothing in between.

## Options

1. **Add a STRICT preset with an in-lane speed cap.** A car's speed is capped while it is still in
   the EV's lane, so it cannot outrun the ambulance in its own lane. EASY and HARD stay as they are.
   **Chosen.**
2. **Shorten the time limit on the existing presets.** Rejected as the main fix: it only closes the
   loophole by making the clock harsher for everyone, which squeezes the learner's margin, and it
   invalidates every published M2 and M3 number.
3. **Raise the blocking penalty.** Rejected: it changes the return but not *success*, so the broken
   measure stays broken.
4. **Change EASY and HARD directly and re-measure everything.** Rejected for now: it would invalidate
   the documented M2-vs-M3 comparison and require retraining every published policy. Still possible if
   we later want one single scenario.
5. **Leave it, and just keep the baseline visible.** Not enough on its own, though the `speedup`
   baseline and its two checks are kept regardless.

## Decision

Add **STRICT**: EASY's road plus `ScenarioConfig.ev_lane_speed_cap` (set to `coop_speed`), which caps a
car's speed **only while it is in the EV's lane**. A car that has moved aside is free to speed up
again, so the cap targets the loophole and not the driving.

Measured over 20 seeds:

| preset | policy | success | `t_clear` | EV speed |
|--------|-------:|--------:|----------:|---------:|
| EASY | `speedup` | 100% | 10.61 s | 4.18 m/s |
| STRICT | `speedup` | **0%** | — | 2.29 m/s (same as `naive`) |
| STRICT | `ideal` | 100% | 6.12 s | 7.31 m/s (unchanged) |

The headroom check gains two STRICT tests, *speeding up cannot succeed* and *moving aside still
succeeds*, so the property is enforced, not assumed. `--preset strict` works in every command that
takes a preset: `clearance-eval`, `clearance-train`, `clearance-train-dec`, `dec-smoke` and
`clearance-watch`. `clearance-smoke` takes no `--preset`; it checks *all* presets in one run, and the
STRICT tests are part of that fixed sequence.

## Consequences

- **Good:** on STRICT, success *means* cooperation. It paid off at once: the per-car policy trained on
  STRICT reaches **exactly** the ideal (100% success, 0 collisions, `t_clear` 6.13 s, 7.296 m/s, return
  102.20, **3.0** yields) and gives **bit-for-bit the same numbers as the central policy**. So M3's
  +9.7% gap on EASY was the learner using the loophole, not a cost of splitting per car. Future
  scenario changes are checked against the `speedup` baseline.
- **Cost:** a third preset to maintain and train on. The cap is a simplification (a real car *could*
  speed up in its lane; it just would not help an ambulance behind it). EASY and HARD keep the
  loophole, so results on them must always be read with the clearance time, not the success rate.
- **Neutral:** the EASY and HARD numbers, models and the published M2-vs-M3 comparison are untouched by
  design.
