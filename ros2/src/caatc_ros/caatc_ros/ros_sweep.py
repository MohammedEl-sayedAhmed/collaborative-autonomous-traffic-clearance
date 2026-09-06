"""M4.4: the two measured columns, loss and delay, through the relay on the real bus.

Everything up to here is a faithfulness check: with a perfect radio the ROS fleet equals
the headless run. This runs the fleet in lockstep with an IMPERFECT radio, several
settings of message loss and of delay, several seeds each, and writes the outcomes as
a table. These are results, not checks: nothing here is required to match anything.

Run inside the caatc-ros image::

    python3 -m caatc_ros.ros_sweep [--preset strict] [--seeds 0,1] [--policy numpy:...]
                                   [--losses 0,0.1,0.3,0.5] [--delays 0,2,5,10] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

from caatc.ros_bridge_core import Record, record_glob, record_metrics
from caatc.scenario import preset_config

from .ros_smoke import FleetOptions, prepare_out_dir, run_lockstep

import glob


def one_setting(preset: str, seeds: List[int], policy: str, loss: float, delay: int, out_dir: str,
                domain: int, timeout_s: float) -> dict:
    d = os.path.join(out_dir, f"loss{loss:g}-delay{delay}")
    prepare_out_dir(d, keep=False, preset=preset)
    cfg = preset_config(preset)
    fleet = FleetOptions(cars=list(range(1, cfg.num_cooperators + 1)), v2v=True, policy=policy,
                        loss=loss, delay_ticks=delay)
    run = run_lockstep(preset, seeds, d, fleet, timeout_s, domain)
    recs = [Record.load(p) for p in sorted(glob.glob(os.path.join(d, record_glob(preset))))]
    ms = [record_metrics(r) for r in recs if not r.meta.get("aborted")]
    row = dict(loss=loss, delay_ticks=delay, seeds=len(ms), bridge_exit=run.bridge_code,
               success=float(np.mean([m["success"] for m in ms])) if ms else None,
               collisions=float(np.mean([m["collision"] for m in ms])) if ms else None,
               t_clear=float(np.mean([m["t_clear"] for m in ms if m["t_clear"] is not None])) if any(m["t_clear"] is not None for m in ms) else None,
               yields=float(np.mean([m["lane_changes"] for m in ms])) if ms else None,
               ev_speed=float(np.mean([m["ev_mean_speed"] for m in ms])) if ms else None,
               ret=float(np.mean([m["cum_reward"] for m in ms])) if ms else None)
    return row


def table(rows: List[dict], preset: str, policy: str, seeds: List[int]) -> str:
    def f(v, fmt):
        return "—" if v is None else format(v, fmt)
    out = [f"## Loss and delay through the relay ({preset}, {policy}, seeds {seeds}, lockstep)", "",
           "| loss | delay (ticks) | success | collisions | mean `t_clear` | EV speed | yields | return |",
           "|-----:|--------------:|--------:|-----------:|---------------:|---------:|-------:|-------:|"]
    for r in rows:
        if not r["seeds"]:
            out.append(f"| {r['loss']:g} | {r['delay_ticks']} | aborted (bridge exit {r['bridge_exit']}) | — | — | — | — | — |")
            continue
        out.append(f"| {r['loss']:g} | {r['delay_ticks']} | {f(r['success'], '.0%')} | {f(r['collisions'], '.0%')} | "
                   f"{f(r['t_clear'], '.2f')} s | {f(r['ev_speed'], '.2f')} m/s | {f(r['yields'], '.1f')} | {f(r['ret'], '.1f')} |")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M4.4: loss and delay sweeps through the relay.")
    ap.add_argument("--preset", default="strict", choices=["easy", "hard", "strict"])
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--policy", default="numpy:caatc/policies/ippo-strict")
    ap.add_argument("--losses", default="0,0.1,0.3,0.5")
    ap.add_argument("--delays", default="0,2,5,10")
    ap.add_argument("--out-dir", default="/src/saved_variables/ros/sweep")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--domain", type=int, default=None)
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    losses = [float(x) for x in a.losses.split(",")]
    delays = [int(x) for x in a.delays.split(",")]
    domain = a.domain if a.domain is not None else 1 + os.getpid() % 100
    os.makedirs(a.out_dir, exist_ok=True)
    settings = [(l, 0) for l in losses] + [(0.0, d) for d in delays if d != 0]
    rows = []
    t0 = time.monotonic()
    for loss, delay in settings:
        print(f"\n=== loss {loss:g}, delay {delay} ticks ===")
        rows.append(one_setting(a.preset, seeds, a.policy, loss, delay, a.out_dir, domain, a.timeout))
        r = rows[-1]
        print(f"  -> success {r['success']}, t_clear {r['t_clear']}, yields {r['yields']} ({r['seeds']} seeds)")
    md = table(rows, a.preset, a.policy, seeds)
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n" + md)
    print(f"({time.monotonic() - t0:.0f} s) -> {a.out_dir}/results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
