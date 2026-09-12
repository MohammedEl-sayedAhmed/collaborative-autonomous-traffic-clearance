"""M5.0 spike: step Gazebo Harmonic headless from Python and measure it.

Two questions the plan (docs/design/m5-3d-plant.md) says must be answered before anything
else is built:

1. **Speed.** How many simulated seconds per wall second does a paused world stepped 10 ms at a
   time reach on this machine, with one car and with four, each with a GPU lidar?
2. **Repeatability.** Run the same drive twice: how far apart do the two trajectories end up?

It drives every car with the same (steer, speed) command through Gazebo's Ackermann system,
steps the world through its control service exactly as the bridge will, and reads the poses
back. No ROS, no caatc code: this is only about the plant.

Two things learned the hard way here. With the Gazebo Python bindings, a blocking service
request and a subscription callback must not live in the same process (the request starves the
callback thread and times out). And a command published on a topic can land one physics step
late relative to the step request. Both went away with the lockstep plugin
(gazebo/plugins/lockstep): commands in and state out through services, nothing subscribed.

Run inside the caatc-gazebo image (run.sh gazebo-spike)::

    python3 gazebo/spike.py [--world spike1|spike4] [--ticks 300] [--repeat 2] [--speed 2.0] [--steer 0.1]
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

WORLD = "spike"
TICK_NS = 10_000_000          # one referee tick = 10 ms of simulated time = ten 1 ms physics steps


def start_server(world_file: str) -> subprocess.Popen:
    """The Gazebo server only, paused, rendering without a screen (EGL)."""
    cmd = ["gz", "sim", "-s", "--headless-rendering", "-v", "1", world_file]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def one_run(a, world_file, cars, run):
    from gz.transport13 import Node
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.double_v_pb2 import Double_V
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.world_control_pb2 import WorldControl

    srv = start_server(world_file)
    node = Node()
    probe = WorldControl(); probe.pause = True
    t0 = time.monotonic()
    while time.monotonic() - t0 < 60:
        ok, _ = node.request(f"/world/{WORLD}/control", probe, WorldControl, Boolean, 500)
        if ok:
            break
        if srv.poll() is not None:
            print("the Gazebo server exited early:\n" + (srv.stdout.read() if srv.stdout else "")); return None
    else:
        print("the Gazebo server did not come up in 60 s"); srv.kill(); return None
    while True:
        ok, rep = node.request("/caatc/lockstep/state", Empty(), Empty, Double_V, 500)
        if ok and len(rep.data) >= 3 and int(rep.data[2]) == len(cars):
            break
        if time.monotonic() - t0 > 60:
            print("the lockstep plugin did not report the cars"); srv.kill(); return None
        time.sleep(0.02)
    boot = time.monotonic() - t0
    iterations = int(rep.data[0])

    cmd = Double_V()
    for _c in cars:
        cmd.data.append(a.steer); cmd.data.append(a.speed)
    step = WorldControl(); step.pause = True; step.multi_step = 10
    traj, waits, req_ms = [], [], []
    wall0 = time.monotonic()
    for t in range(a.ticks):
        r0 = time.monotonic()
        ok, _ = node.request("/caatc/lockstep/command", cmd, Double_V, Boolean, 500)
        ok2, _ = node.request(f"/world/{WORLD}/control", step, WorldControl, Boolean, 500)
        req_ms.append(1e3 * (time.monotonic() - r0))
        if not (ok and ok2):
            print(f"tick {t}: a service did not answer"); srv.kill(); return None
        iterations += 10
        w0 = time.monotonic()
        while True:
            ok, rep = node.request("/caatc/lockstep/state", Empty(), Empty, Double_V, 500)
            if ok and int(rep.data[0]) >= iterations:
                break
            if time.monotonic() - w0 > 5.0:
                print(f"tick {t}: the world did not reach iteration {iterations} in 5 s"); srv.kill(); return None
            time.sleep(0.0002)
        waits.append(time.monotonic() - w0)
        traj.append((rep.data[3], rep.data[4]))
    wall = time.monotonic() - wall0
    sim_t = float(rep.data[1])
    final = {c: (rep.data[3 + 6 * k], rep.data[4 + 6 * k], 0.05) for k, c in enumerate(cars)}
    srv.kill(); srv.wait()
    del node
    time.sleep(0.3)
    tr = np.array(traj)
    path = float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1))) if len(tr) > 1 else 0.0
    last = tr[-101:] if len(tr) > 101 else tr
    v_last = float(np.sum(np.linalg.norm(np.diff(last, axis=0), axis=1))) / (0.01 * max(1, len(last) - 1))
    res = dict(run=run, boot_s=boot, wall_s=wall, sim_s=a.ticks * 0.01, rtf=(a.ticks * 0.01) / wall, path_m=path, v_last=v_last,
               mean_tick_ms=1e3 * wall / a.ticks, mean_wait_ms=1e3 * float(np.mean(waits)), p99_wait_ms=1e3 * float(np.percentile(waits, 99)),
               mean_req_ms=float(np.mean(req_ms)), retries=0, lidar_msgs=-1, sim_t=sim_t, final=final)
    print(f"run {run}: server up in {boot:.1f} s; {a.ticks} ticks ({a.ticks * 0.01:.2f} sim s) in {wall:.2f} wall s -> "
          f"real-time factor {res['rtf']:.2f}; per tick {res['mean_tick_ms']:.2f} ms (the two requests {res['mean_req_ms']:.2f} ms, "
          f"physics+sensors wait mean {res['mean_wait_ms']:.2f} ms, p99 {res['p99_wait_ms']:.2f} ms); sim clock {sim_t:.3f} s")
    print(f"   the first car: path {path:.3f} m in {a.ticks * 0.01:.2f} s; speed over the last second {v_last:.2f} m/s (commanded {a.speed:.2f} m/s)")
    for c in cars:
        p = final[c]
        print(f"   {c}: x={p[0]:.6f} y={p[1]:.6f}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="spike1", help="spike1 (one car) or spike4 (four cars)")
    ap.add_argument("--ticks", type=int, default=300, help="10 ms ticks per run (300 = 3 simulated s)")
    ap.add_argument("--repeat", type=int, default=2, help="how many identical runs, for repeatability")
    ap.add_argument("--speed", type=float, default=2.0, help="m/s, every car")
    ap.add_argument("--steer", type=float, default=0.0, help="rad, every car (0 = straight between the walls; 0.1 hits the wall after 1.5 s)")
    ap.add_argument("--wheelbase", type=float, default=0.3302)
    a = ap.parse_args(argv)
    try:
        import gz.transport13, gz.msgs10  # noqa: F401
    except ImportError as e:
        print(f"the Gazebo Python bindings are missing: {e}"); return 2

    world_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlds", f"{a.world}.sdf")
    n_cars = 1 if a.world.endswith("1") else 4
    cars = [f"car{i}" for i in range(1, n_cars + 1)]
    results = []
    for run in range(a.repeat):
        r = one_run(a, world_file, cars, run)
        if r is None:
            return 4
        results.append(r)

    if len(results) >= 2:
        worst = 0.0
        for c in cars:
            ps = [np.array(r["final"][c]) for r in results]
            worst = max(worst, float(max(np.linalg.norm(p - ps[0]) for p in ps[1:])))
        print(f"\nrepeatability: the largest difference in a car's final position across {len(results)} identical runs: "
              f"{worst:.6f} m ({'bit-identical' if worst == 0.0 else 'not identical'})")
    print(f"the first car drove {results[0]['path_m']:.2f} m of path in {results[0]['sim_s']:.2f} s and reached "
          f"{results[0]['v_last']:.2f} m/s of the {a.speed:.2f} m/s commanded: the vehicle model's response is M5.1's business")
    rtfs = [r["rtf"] for r in results]
    print(f"speed: real-time factor {min(rtfs):.2f} to {max(rtfs):.2f} with {n_cars} car(s), each with a 1081-point GPU lidar at 40 Hz (rendering, not subscribed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
