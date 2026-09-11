"""M5.1: the referee on the Gazebo plant, headless, no ROS.

Runs ``ClearanceEnv`` with ``GazeboPlant`` for a preset, some seeds and some policies, and
writes a results table. Three of the plan's checks live here (docs/design/m5-3d-plant.md):

* **repeatability (check 3):** the first seed and policy run twice; the largest difference in
  any car's state at any decision step is printed (the spike measured 0);
* **headroom (check 7):** on STRICT ``naive`` must fail and ``ideal`` must succeed on every seed;
* **the tables (check 8):** success, collisions, ``t_clear``, EV speed, yields, return per
  policy, plus the plant's real-time factor (check 10). Nothing is required to match the 2D
  plant: a difference is the result.

Run inside caatc-gazebo (``./run.sh gazebo-smoke``)::

    python3 -m caatc.gazebo_smoke --preset strict --seeds 0,1 --policies ideal,naive,numpy:caatc/policies/ippo-strict
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from .clearance_env import ClearanceEnv
from .gazebo_plant import GazeboPlant
from .ros_bridge_core import cars_to_array
from .ros_node_core import policy_from_name
from .scenario import ScenarioConfig, preset_config

POLICY_ALIAS = {"ideal": "local-ideal"}


class Squad:
    """The same per-car policy on every cooperator, each reading only its own 26 numbers."""

    def __init__(self, cfg: ScenarioConfig, name: str):
        self.cfg = cfg
        self.policies = [policy_from_name(POLICY_ALIAS.get(name, name)) for _ in range(cfg.num_cooperators)]

    def __call__(self, env: ClearanceEnv) -> np.ndarray:
        obs = env.per_agent_obs_all()
        return np.array([int(p(obs[j], self.cfg)) for j, p in enumerate(self.policies)], dtype=int)


def run_one(cfg: ScenarioConfig, plant: GazeboPlant, policy_name: str, seed: int) -> Dict:
    env = ClearanceEnv(cfg, plant=plant)
    squad = Squad(cfg, policy_name)
    try:
        t0 = time.monotonic()
        _obs, info = env.reset(seed=seed)
        states = [cars_to_array(env.cars)]
        total, speeds, trace = 0.0, [], []
        while True:
            _obs, r, term, trunc, info = env.step(squad(env))
            total += float(r); speeds.append(float(info["ev_v"]))
            states.append(cars_to_array(env.cars))
            ev = env.cars[0]
            trace.append((info["sim_time"], ev["s"], ev["d"], ev["v"], ev["delta"], float(plant.last_rows[0, 0]), float(plant.last_rows[0, 1])))
            if term or trunc:
                break
        wall = time.monotonic() - t0
        contacts = plant.contacts()
        return dict(policy=policy_name, seed=seed, success=bool(info["success"]), collision=bool(info["collision"]),
                    contacts=contacts, contacts_summary=sorted({" ".join(l.split()[1:]) for l in contacts}),
                    first_contact_s=(float(contacts[0].split()[0]) if contacts else None),
                    t_clear=info["t_clear"], ev_mean_speed=float(np.mean(speeds)), lane_changes=int(info["lane_changes"]),
                    cum_reward=total, steps=len(states) - 1, sim_s=float(info["sim_time"]), wall_s=wall,
                    rtf=float(info["sim_time"]) / wall if wall > 0 else None, states=np.stack(states),
                    plant_rtf=plant.real_time_factor(), late_reads=plant.stats["late_reads"], retries=plant.stats["retries"],
                    late_commands=plant.stats["late_commands"], trace=np.array(trace))
    finally:
        env.close()


def fmt(v, spec):
    return "—" if v is None else format(v, spec)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="strict", choices=["easy", "hard", "strict"])
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--policies", default="ideal,naive,numpy:caatc/policies/ippo-strict")
    ap.add_argument("--repeatability", type=int, default=2, help="how many times the first seed+policy runs; 0 skips")
    ap.add_argument("--out-dir", default="/src/saved_variables/gazebo")
    ap.add_argument("--no-walls", action="store_true", help="a world without the lidar walls")
    ap.add_argument("--trace", action="store_true", help="print the emergency vehicle's trace of the first run")
    a = ap.parse_args(argv)
    cfg = preset_config(a.preset)
    seeds = [int(s) for s in a.seeds.split(",") if s]
    policies = [p for p in a.policies.split(",") if p]
    os.makedirs(a.out_dir, exist_ok=True)
    plant = GazeboPlant(cfg, walls=not a.no_walls)
    ok = True
    rows: List[Dict] = []
    print(f"=== the referee on the Gazebo plant: preset={a.preset} seeds={seeds} policies={policies} ===")
    try:
        if a.repeatability >= 2:
            print(f"\nCheck 3: repeatability ({policies[0]}, seed {seeds[0]}, {a.repeatability} identical runs)")
            runs = [run_one(cfg, plant, policies[0], seeds[0]) for _ in range(a.repeatability)]
            same_len = len({r["steps"] for r in runs}) == 1
            worst, first_diff = 0.0, None
            for r in runs[1:]:
                m = min(len(r["states"]), len(runs[0]["states"]))
                d = np.abs(r["states"][:m] - runs[0]["states"][:m])
                worst = max(worst, float(d.max()))
                bad = np.where(d.reshape(m, -1).max(axis=1) > 0)[0]
                if bad.size and (first_diff is None or int(bad[0]) < first_diff):
                    first_diff = int(bad[0])
            outcomes = {(r["success"], r["collision"], r["t_clear"], r["lane_changes"]) for r in runs}
            print(f"    steps {[r['steps'] for r in runs]}; largest state difference on the common prefix {worst:.9f}"
                  f"{'' if first_diff is None else f', first at decision step {first_diff}'} "
                  f"({'bit-identical' if same_len and worst == 0.0 else 'NOT identical'}); outcomes identical: {len(outcomes) == 1}")
            for r in runs:
                print(f"    run: {r['steps']} steps, collision={r['collision']}, first contact at {fmt(r['first_contact_s'], '.2f')} s, "
                      f"late reads {r['late_reads']}, late commands {r['late_commands']}, retries {r['retries']}; contacts: {r['contacts_summary'][:6]}")
            if first_diff is not None:
                a0, a1 = runs[0]["states"][first_diff], runs[1]["states"][first_diff]
                fields = ("x", "y", "theta", "v", "delta", "s", "d", "lane")
                bad = [(i, fields[k], float(a0[i, k]), float(a1[i, k])) for i in range(a0.shape[0]) for k in range(a0.shape[1]) if a0[i, k] != a1[i, k]]
                print(f"    first difference (step {first_diff}): " + "; ".join(f"car{i} {f}: {u:.6g} vs {w:.6g}" for i, f, u, w in bad[:6]))
            if a.trace:
                tr = runs[0]["trace"]
                print("    EV trace (sim s, s, d, v, delta achieved, steer commanded, speed commanded), every 5th step:")
                for row in tr[::5]:
                    print("      " + " ".join(f"{v:8.3f}" for v in row))
                print(f"    EV: max |d| {np.max(np.abs(tr[:, 2])):.3f} m; max |steer cmd - delta| {np.max(np.abs(tr[:, 5] - tr[:, 4])):.3f} rad")
            rows.extend(runs)
        print("\nChecks 7, 8 and 10: the policies on every seed")
        for pol in policies:
            for seed in seeds:
                if any(r["policy"] == pol and r["seed"] == seed for r in rows):
                    continue
                rows.append(run_one(cfg, plant, pol, seed))
                r = rows[-1]
                print(f"    {pol:>34} seed {seed}: success={r['success']} collision={r['collision']} t_clear={fmt(r['t_clear'], '.2f')} "
                      f"yields={r['lane_changes']} return={r['cum_reward']:.1f} ({r['sim_s']:.2f} sim s in {r['wall_s']:.1f} s, "
                      f"plant real-time factor {fmt(r['plant_rtf'], '.2f')}, late reads {r['late_reads']})")
        if a.preset == "strict":
            for pol, want in (("naive", False), ("ideal", True)):
                got = [r for r in rows if r["policy"] == pol]
                if got:
                    passed = all(r["success"] == want for r in got)
                    ok &= passed
                    print(f"    [{'PASS' if passed else 'FAIL'}] headroom carries over: {pol} {'succeeds' if want else 'fails'} on every seed")
    finally:
        plant.close()

    # the table, one line per policy (means over seeds; the repeat runs count once)
    seen = set(); uniq = []
    for r in rows:
        k = (r["policy"], r["seed"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    lines = [f"## The referee on the Gazebo plant ({a.preset}, seeds {seeds}, lockstep, ground-truth poses)", "",
             "| policy | success | collisions | mean `t_clear` | EV speed | yields | return | plant real-time factor |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for pol in policies:
        g = [r for r in uniq if r["policy"] == pol]
        if not g:
            continue
        tc = [r["t_clear"] for r in g if r["t_clear"] is not None]
        lines.append(f"| {pol} | {np.mean([r['success'] for r in g]):.0%} | {np.mean([r['collision'] for r in g]):.0%} | "
                     f"{fmt(float(np.mean(tc)) if tc else None, '.2f')} s | {np.mean([r['ev_mean_speed'] for r in g]):.2f} m/s | "
                     f"{np.mean([r['lane_changes'] for r in g]):.1f} | {np.mean([r['cum_reward'] for r in g]):.1f} | "
                     f"{np.mean([r['plant_rtf'] for r in g if r['plant_rtf']]):.2f} |")
    table = "\n".join(lines) + "\n"
    with open(os.path.join(a.out_dir, f"results-{a.preset}.md"), "w") as f:
        f.write(table)
    with open(os.path.join(a.out_dir, f"results-{a.preset}.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("states", "contacts", "trace")} for r in rows], f, indent=1, default=float)
    print("\n" + table)
    print(("OK" if ok else "FAIL") + f": results in {a.out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
