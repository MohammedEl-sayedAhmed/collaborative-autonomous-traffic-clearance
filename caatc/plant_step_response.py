"""M5.1 check 6: a steering step and a speed step on each plant, side by side.

Not pass/fail: a table. One car (the emergency vehicle's slot, car 0) is driven alone on a
straight command while every other car is told to stay still. Two experiments:

* **speed step:** steer 0, speed 0 -> 4 m/s at t = 0. Rise time to 90%, overshoot, steady error.
* **steering step:** speed held at 2 m/s, steer 0 -> 0.3 rad at t = 1 s. Rise time of the
  steering angle to 90%, overshoot, and the steady yaw rate (a proxy for what the steering does).

Run inside caatc-gazebo (``./run.sh gazebo-step-response``)::

    python3 -m caatc.plant_step_response [--plants gym,gazebo] [--out saved_variables/gazebo/step-response.md]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np

from .plants import make_plant
from .scenario import easy_preset

TICK = 0.01


def _run(plant, n: int, ticks: int, command) -> Dict[str, np.ndarray]:
    poses = np.zeros((n, 3)); poses[:, 0] = 10.0 - 3.0 * np.arange(n)     # a line of cars, well apart, heading +x
    st = plant.reset(poses)
    t, v, delta, theta = [0.0], [float(st.v[0])], [float(st.delta[0])], [float(st.theta[0])]
    for k in range(ticks):
        rows = np.zeros((n, 2))
        rows[0] = command((k + 1) * TICK)
        st, _ = plant.substep(rows)
        t.append((k + 1) * TICK); v.append(float(st.v[0])); delta.append(float(st.delta[0])); theta.append(float(st.theta[0]))
    return dict(t=np.array(t), v=np.array(v), delta=np.array(delta), theta=np.array(theta))


def _step_stats(t: np.ndarray, y: np.ndarray, t0: float, target: float) -> Dict[str, float]:
    m = t >= t0
    tt, yy = t[m] - t0, y[m]
    steady = float(np.mean(yy[-50:]))
    rise = tt[np.argmax(np.abs(yy) >= 0.9 * abs(target))] if np.any(np.abs(yy) >= 0.9 * abs(target)) else float("nan")
    overshoot = max(0.0, (np.max(np.abs(yy)) - abs(target)) / abs(target) * 100.0)
    return dict(rise_s=float(rise), overshoot_pct=float(overshoot), steady=steady, steady_error=steady - target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plants", default="gym,gazebo")
    ap.add_argument("--out", default="/src/saved_variables/gazebo/step-response.md")
    a = ap.parse_args(argv)
    cfg = easy_preset()
    n = cfg.num_agents
    rows: List[str] = []
    for name in a.plants.split(","):
        plant = make_plant(name, cfg)
        if plant is None:
            from .plant import GymPlant
            from .scenario import build_track
            plant = GymPlant(cfg, build_track(cfg))
        try:
            t0 = time.monotonic()
            sp = _run(plant, n, 300, lambda t: (0.0, 4.0))
            ss = _run(plant, n, 300, lambda t: (0.3 if t >= 1.0 else 0.0, 2.0))
            wall = time.monotonic() - t0
        finally:
            plant.close()
        s1 = _step_stats(sp["t"], sp["v"], 0.0, 4.0)
        s2 = _step_stats(ss["t"], ss["delta"], 1.0, 0.3)
        m = ss["t"] >= 2.0
        yaw_rate = float(np.mean(np.diff(np.unwrap(ss["theta"][m])) / TICK))
        expected_yaw = 2.0 * math.tan(0.3) / 0.3302
        rows.append(f"| {name} | {s1['rise_s']:.2f} s | {s1['overshoot_pct']:.1f}% | {s1['steady_error']:+.3f} m/s | "
                    f"{s2['rise_s']:.2f} s | {s2['overshoot_pct']:.1f}% | {s2['steady_error']:+.3f} rad | {yaw_rate:.2f} rad/s "
                    f"(kinematic {expected_yaw:.2f}) | {wall:.1f} s |")
        print(rows[-1])
    table = ("## Step responses of the two plants (one car alone, check 6; measured, not required)\n\n"
             "| plant | speed rise to 3.6 m/s | speed overshoot | speed steady error | steer rise to 0.27 rad | steer overshoot | steer steady error | yaw rate at 2 m/s, 0.3 rad | wall time |\n"
             "|---|---:|---:|---:|---:|---:|---:|---|---:|\n" + "\n".join(rows) + "\n")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(table)
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
