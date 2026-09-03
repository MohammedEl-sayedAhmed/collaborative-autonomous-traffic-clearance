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
from typing import List, Optional

import numpy as np

from .scenario import easy_preset, hard_preset
from .clearance_env import ClearanceEnv
from .clearance_eval import run_episode
from .baselines import NaiveHold, RandomPolicy, IdealCooperator
from .decentralized import LocalOnlyView, LocalSquad

# thresholds
SPEED_RATIO_MIN = 3.0
NAIVE_SUCCESS_MAX = 0.05
IDEAL_SUCCESS_MIN = 0.95
PROGRESS_GAP_MIN = 0.20        # ideal_progress - naive_progress
HARD_RANDOM_COLLISION_MIN = 0.30
MONO_EPS = 0.02                # tolerance for "non-decreasing"
LOCAL_SUCCESS_MIN = 0.95       # the local oracle must essentially always succeed
LOCAL_TCLEAR_TOL = 0.10        # ... within 10% of the privileged oracle's t_clear


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
        np.mean([r["cum_reward"] for r in records]) if records else 0.0,
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
    n_succ, _n_coll, n_spd, n_prog, n_ret = _rates(naive)
    i_succ, _i_coll, i_spd, i_prog, i_ret = _rates(ideal)
    ratio = i_spd / n_spd if n_spd > 1e-6 else float("inf")
    print(f"    naive: success={n_succ:.2f} ev_speed={n_spd:.2f} progress={n_prog:.2f} return={n_ret:.2f}")
    print(f"    ideal: success={i_succ:.2f} ev_speed={i_spd:.2f} progress={i_prog:.2f} return={i_ret:.2f}")
    g.check("naive truncates (success ~ 0)", n_succ <= NAIVE_SUCCESS_MAX, f"{n_succ:.2f}")
    g.check("ideal succeeds (100%)", i_succ >= IDEAL_SUCCESS_MIN, f"{i_succ:.2f}")
    g.check("EV speed ratio ideal/naive >= 3", ratio >= SPEED_RATIO_MIN, f"{ratio:.2f}x")
    g.check("progress gap ideal-naive >= 0.2", i_prog - n_prog >= PROGRESS_GAP_MIN,
            f"{i_prog - n_prog:.2f}")
    # guard the training signal itself: the cooperative reward must rank ideal above
    # naive (a sign/weight bug would otherwise pass every physics-based check).
    g.check("cooperative return ideal > naive", i_ret > n_ret, f"{i_ret:.1f} > {n_ret:.1f}")

    # -- 3. monotonicity -----------------------------------------------------
    print(f"\nMonotonicity ({mono_seeds} seeds/point):")
    progs = []
    for m in range(cfg.num_cooperators + 1):
        recs = [run_episode(env, IdealCooperator(yield_count=m), seed=s) for s in range(mono_seeds)]
        _s, _c, spd, prog, _r = _rates(recs)
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
    _rs, r_coll, _rspd, _rprog, _rret = _rates(rnd)
    _is, i_coll, _ispd, i_prog2, _iret = _rates(idl)
    print(f"    random: collision={r_coll:.2f}")
    print(f"    ideal:  collision={i_coll:.2f} progress={i_prog2:.2f}")
    g.check("HARD random collides substantially", r_coll >= HARD_RANDOM_COLLISION_MIN, f"{r_coll:.2f}")
    g.check("HARD ideal never collides", i_coll == 0.0, f"{i_coll:.2f}")
    envh.close()

    return g.passed()


def run_m3_gate(seeds: int = 20, num_cooperators: Optional[int] = None) -> bool:
    """M3 gate: is the per-agent observation SUFFICIENT, with no global state?

    Deploys the scripted **local** oracle -- which sees only ``per_agent_obs(j)`` --
    to all K cars, through the same ``run_episode`` path as every other policy, and
    requires it to match the privileged oracle. If a hand-written local policy can
    do the job, learning it is a tractable problem; if it cannot, the observation is
    missing information and no amount of training fixes that (ADR 0009).

    Also checks that the executor runs against ``LocalOnlyView``, i.e. that it
    cannot even reach joint state.
    """
    g = Gate()
    over = {} if num_cooperators is None else {"num_cooperators": num_cooperators}

    for label, cfg in (("EASY", easy_preset(**over)), ("HARD", hard_preset(**over))):
        env = ClearanceEnv(cfg)
        try:
            priv = [run_episode(env, IdealCooperator(), seed=s) for s in range(seeds)]
            loc = [run_episode(env, LocalSquad(), seed=s) for s in range(seeds)]
        finally:
            env.close()
        p_succ, p_coll, _p_spd, _p_prog, _p_ret = _rates(priv)
        l_succ, l_coll, l_spd, _l_prog, _l_ret = _rates(loc)
        p_tc = np.mean([r["t_clear"] for r in priv if r["t_clear"]]) if any(
            r["t_clear"] for r in priv) else None
        l_tc = np.mean([r["t_clear"] for r in loc if r["t_clear"]]) if any(
            r["t_clear"] for r in loc) else None
        print(f"\n{label} ({seeds} seeds, K={cfg.num_cooperators}):")
        print(f"    privileged oracle: success={p_succ:.2f} coll={p_coll:.2f} "
              f"t_clear={('%.2f' % p_tc) if p_tc else '-'}")
        print(f"    local oracle:      success={l_succ:.2f} coll={l_coll:.2f} "
              f"t_clear={('%.2f' % l_tc) if l_tc else '-'} speed={l_spd:.2f}")
        g.check(f"{label}: local oracle succeeds", l_succ >= LOCAL_SUCCESS_MIN, f"{l_succ:.2f}")
        g.check(f"{label}: local oracle never collides", l_coll == 0.0, f"{l_coll:.2f}")
        if p_tc and l_tc:
            rel = abs(l_tc - p_tc) / p_tc
            g.check(f"{label}: local t_clear within 10% of privileged",
                    rel <= LOCAL_TCLEAR_TOL, f"{rel * 100:.1f}%")

    # the executor must work with NO access to joint state
    env = ClearanceEnv(easy_preset(**over))
    try:
        env.reset(seed=0)
        action = LocalSquad()(LocalOnlyView(env))
        ok_shape = np.asarray(action).shape == (env.cfg.num_cooperators,)
        leaked = True
        try:
            LocalOnlyView(env)._cars          # must raise
            leaked = True
        except AttributeError:
            leaked = False
    finally:
        env.close()
    print()
    g.check("executor acts through a local-only view", ok_shape)
    g.check("the local-only view hides joint state", not leaked)
    return g.passed()


def main(argv=None):
    ap = argparse.ArgumentParser(description="ClearanceEnv pre-training headroom gate.")
    ap.add_argument("--easy-seeds", type=int, default=10)
    ap.add_argument("--hard-seeds", type=int, default=8)
    ap.add_argument("--mono-seeds", type=int, default=3)
    ap.add_argument("--m3", action="store_true",
                    help="run the M3 observability gate (the local oracle) instead")
    ap.add_argument("--num-cooperators", type=int, default=None,
                    help="override K, so any K that M3 claims is gated before it is evaluated")
    a = ap.parse_args(argv)

    if a.m3:
        print("=== M3 observability gate: is the per-agent view sufficient? ===")
        ok = run_m3_gate(seeds=a.easy_seeds, num_cooperators=a.num_cooperators)
        print("\n" + ("OK: the local view is sufficient -- decentralization is well posed."
                      if ok else
                      "FAIL: the local view is NOT sufficient -- enrich the observation "
                      "before training a decentralized policy."))
        return 0 if ok else 1

    print("=== ClearanceEnv headroom gate (headless) ===")
    ok = run_gate(a.easy_seeds, a.hard_seeds, a.mono_seeds)
    print("\n" + ("OK: headroom confirmed -- safe to train." if ok else
                  "FAIL: no headroom -- do NOT train on this scenario."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
