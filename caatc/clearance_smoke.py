"""Pre-training headroom gate for ClearanceEnv.

Proves -- headless, in seconds, before any training is spent -- that the scenario
has real headroom: a naive policy is clearly worse than a cooperative one *by
construction*, so a learner has a dense gradient to climb (the legacy toy harness
taught us that a saturated scenario, where even random ~ optimal, hides every
algorithm difference).

Checks (see ``docs/design/m1-clearance-env.md`` -- "Verification"):
  1. Static math: naive is clamped to ``coop_speed`` (t ~ s_goal/coop_speed),
     ideal sprints at ``ev_max`` (t ~ s_goal/ev_max); ratio >= 3; ``max_time``
     sits between so naive truncates and ideal succeeds.
  2. EASY empirical: naive success ~ 0 (truncates), ideal success = 100%, and the
     EV speed ratio ideal/naive >= 3.
  3. Monotonicity: yielding the first ``m`` cooperators gives EV progress that
     increases with ``m`` (dense partial credit; random << optimal, optimum
     reachable).
  4. HARD: random collides substantially while ideal never collides.

Exits non-zero if any check fails -- training is never spent on a secretly
saturated scenario.
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np

from .scenario import easy_preset, hard_preset
from .clearance_env import ClearanceEnv
from .clearance_eval import run_episode
from .baselines import NaiveHold, RandomPolicy, IdealCooperator

# thresholds
SPEED_RATIO_MIN = 3.0
NAIVE_SUCCESS_MAX = 0.05
IDEAL_SUCCESS_MIN = 0.95
PROGRESS_GAP_MIN = 0.20        # ideal_progress - naive_progress
HARD_RANDOM_COLLISION_MIN = 0.30
MONO_EPS = 0.02                # tolerance for "non-decreasing"


class Gate:
    def __init__(self):
        self.checks: List[tuple] = []

    def check(self, name: str, ok: bool, detail: str = ""):
        self.checks.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
        return ok

    def passed(self) -> bool:
        return all(ok for _n, ok, _d in self.checks)


def _rates(records):
    n = max(1, len(records))
    return (
        sum(r["success"] for r in records) / n,
        sum(r["collision"] for r in records) / n,
        np.mean([r["ev_mean_speed"] for r in records]) if records else 0.0,
        np.mean([r["ev_progress"] for r in records]) if records else 0.0,
    )


def run_gate(easy_seeds: int = 10, hard_seeds: int = 8, mono_seeds: int = 3) -> bool:
    g = Gate()

    # -- 1. static math ------------------------------------------------------
    cfg = easy_preset()
    t_naive = cfg.s_goal / cfg.coop_speed
    t_ideal = cfg.s_goal / cfg.ev_max_speed
    print("Static math:")
    g.check("speed ratio >= 3 (ev_max / coop)", cfg.ev_max_speed / cfg.coop_speed >= 3.0,
            f"{cfg.ev_max_speed / cfg.coop_speed:.1f}x")
    g.check("T_max between naive and ideal clear times",
            t_ideal < cfg.max_time < t_naive,
            f"t_ideal={t_ideal:.1f}s < T_max={cfg.max_time:.1f}s < t_naive={t_naive:.1f}s")

    # -- 2. EASY empirical ---------------------------------------------------
    print(f"\nEASY empirical ({easy_seeds} seeds):")
    env = ClearanceEnv(cfg)
    naive = [run_episode(env, NaiveHold(), seed=s) for s in range(easy_seeds)]
    ideal = [run_episode(env, IdealCooperator(), seed=s) for s in range(easy_seeds)]
    n_succ, _n_coll, n_spd, n_prog = _rates(naive)
    i_succ, _i_coll, i_spd, i_prog = _rates(ideal)
    ratio = i_spd / n_spd if n_spd > 1e-6 else float("inf")
    print(f"    naive: success={n_succ:.2f} ev_speed={n_spd:.2f} progress={n_prog:.2f}")
    print(f"    ideal: success={i_succ:.2f} ev_speed={i_spd:.2f} progress={i_prog:.2f}")
    g.check("naive truncates (success ~ 0)", n_succ <= NAIVE_SUCCESS_MAX, f"{n_succ:.2f}")
    g.check("ideal succeeds (100%)", i_succ >= IDEAL_SUCCESS_MIN, f"{i_succ:.2f}")
    g.check("EV speed ratio ideal/naive >= 3", ratio >= SPEED_RATIO_MIN, f"{ratio:.2f}x")
    g.check("progress gap ideal-naive >= 0.2", i_prog - n_prog >= PROGRESS_GAP_MIN,
            f"{i_prog - n_prog:.2f}")

    # -- 3. monotonicity -----------------------------------------------------
    print(f"\nMonotonicity ({mono_seeds} seeds/point):")
    progs = []
    for m in range(cfg.num_cooperators + 1):
        recs = [run_episode(env, IdealCooperator(yield_count=m), seed=s) for s in range(mono_seeds)]
        _s, _c, spd, prog = _rates(recs)
        progs.append(prog)
        print(f"    m={m}: progress={prog:.3f} ev_speed={spd:.2f}")
    non_decreasing = all(progs[m] >= progs[m - 1] - MONO_EPS for m in range(1, len(progs)))
    g.check("EV progress non-decreasing in m", non_decreasing,
            str([round(float(p), 2) for p in progs]))
    g.check("optimum reachable (progress[K] - progress[0] >= 0.2)",
            progs[-1] - progs[0] >= PROGRESS_GAP_MIN, f"{progs[-1] - progs[0]:.2f}")
    env.close()

    # -- 4. HARD -------------------------------------------------------------
    print(f"\nHARD occupancy ({hard_seeds} seeds):")
    cfgh = hard_preset()
    envh = ClearanceEnv(cfgh)
    rnd = [run_episode(envh, RandomPolicy(s), seed=s) for s in range(hard_seeds)]
    idl = [run_episode(envh, IdealCooperator(), seed=s) for s in range(hard_seeds)]
    _rs, r_coll, _rspd, _rprog = _rates(rnd)
    _is, i_coll, _ispd, i_prog2 = _rates(idl)
    print(f"    random: collision={r_coll:.2f}")
    print(f"    ideal:  collision={i_coll:.2f} progress={i_prog2:.2f}")
    g.check("HARD random collides substantially", r_coll >= HARD_RANDOM_COLLISION_MIN, f"{r_coll:.2f}")
    g.check("HARD ideal never collides", i_coll == 0.0, f"{i_coll:.2f}")
    envh.close()

    return g.passed()


def main(argv=None):
    ap = argparse.ArgumentParser(description="ClearanceEnv pre-training headroom gate.")
    ap.add_argument("--easy-seeds", type=int, default=10)
    ap.add_argument("--hard-seeds", type=int, default=8)
    ap.add_argument("--mono-seeds", type=int, default=3)
    a = ap.parse_args(argv)

    print("=== ClearanceEnv headroom gate (headless) ===")
    ok = run_gate(a.easy_seeds, a.hard_seeds, a.mono_seeds)
    print("\n" + ("OK: headroom confirmed -- safe to train." if ok else
                  "FAIL: no headroom -- do NOT train on this scenario."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
