"""M5.0 spike: step Gazebo Harmonic headless from Python and measure it.

Two questions the plan (docs/design/m5-3d-plant.md) says must be answered before anything
else is built:

1. **Speed.** How many simulated seconds per wall second does a paused world stepped 10 ms at a
   time reach on this machine, with one car and with four, each with a GPU lidar?
2. **Repeatability.** Run the same drive twice: how far apart do the two trajectories end up?

It drives every car with the same (steer, speed) command through Gazebo's Ackermann system,
steps the world through its control service exactly as the bridge will, and reads the poses
back. No ROS, no caatc code: this is only about the plant.

One thing learned the hard way, and kept here on purpose: with the Gazebo Python bindings, a
blocking service request and a subscription callback must NOT live in the same process. The
request holds Python's lock while the transport thread needs it to deliver the callback, and
the request then times out (199 of 200 failed in a test; 0 of 200 without a subscription). So
the listener (clock, poses, lidar) is a separate process that writes into shared memory, and
the stepper process only publishes and requests.

Run inside the caatc-gazebo image (run.sh gazebo-spike)::

    python3 gazebo/spike.py [--world spike1|spike4] [--ticks 300] [--repeat 2] [--speed 2.0] [--steer 0.1]
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
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


def listener(cars, sim_ns, poses, traj, lidar_count, ready, stop):
    """Child process: every subscription lives here and writes into shared memory."""
    from gz.transport13 import Node
    from gz.msgs10.clock_pb2 import Clock
    from gz.msgs10.laserscan_pb2 import LaserScan
    from gz.msgs10.pose_v_pb2 import Pose_V

    node = Node()

    def on_clock(msg):
        sim_ns.value = msg.sim.sec * 1_000_000_000 + msg.sim.nsec

    def on_pose(msg, k):
        for p in msg.pose:
            if p.name == cars[k]:
                poses[3 * k] = p.position.x
                poses[3 * k + 1] = p.position.y
                poses[3 * k + 2] = p.position.z
                if k == 0:                                   # the first car's trajectory, one slot per tick
                    st = p.header.stamp.sec * 1_000_000_000 + p.header.stamp.nsec
                    if st == 0:
                        st = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
                    if st == 0:
                        st = sim_ns.value                    # no stamp on the message: the clock we just heard
                    i = st // TICK_NS
                    if 0 <= i < len(traj) // 2:
                        traj[2 * i] = p.position.x
                        traj[2 * i + 1] = p.position.y

    def on_lidar(_msg):
        lidar_count.value += 1

    node.subscribe(Clock, f"/world/{WORLD}/clock", on_clock)                 # every physics step
    for k, c in enumerate(cars):
        node.subscribe(Pose_V, f"/model/{c}/pose", lambda m, k=k: on_pose(m, k))   # 100 Hz, per car
    node.subscribe(LaserScan, f"/world/{WORLD}/model/{cars[0]}/link/chassis/sensor/lidar/scan", on_lidar)
    ready.value = 1
    while not stop.value:
        time.sleep(0.01)


def one_run(a, world_file, cars, run):
    from gz.transport13 import Node
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.twist_pb2 import Twist
    from gz.msgs10.world_control_pb2 import WorldControl

    ctx = mp.get_context("spawn")
    sim_ns, lidar_count, ready, stop = ctx.Value("q", 0), ctx.Value("q", 0), ctx.Value("i", 0), ctx.Value("i", 0)
    poses = ctx.Array("d", [float("nan")] * (3 * len(cars)))
    traj = ctx.Array("d", [float("nan")] * (2 * (a.ticks + 2)))

    srv = start_server(world_file)
    node = Node()
    t0 = time.monotonic()
    probe = WorldControl(); probe.pause = True               # an EMPTY request means "pause: false" and would start the world
    while time.monotonic() - t0 < 60:                       # the control service appears once the world is loaded
        ok, _ = node.request(f"/world/{WORLD}/control", probe, WorldControl, Boolean, 500)
        if ok:
            break
        if srv.poll() is not None:
            print("the Gazebo server exited early:\n" + (srv.stdout.read() if srv.stdout else "")); return None
    else:
        print("the Gazebo server did not come up in 60 s"); srv.kill(); return None
    boot = time.monotonic() - t0

    lis = ctx.Process(target=listener, args=(cars, sim_ns, poses, traj, lidar_count, ready, stop), daemon=True)
    lis.start()
    while not ready.value:
        time.sleep(0.01)
    time.sleep(0.5)                                          # let discovery settle
    pubs = {c: node.advertise(f"/model/{c}/cmd_vel", Twist) for c in cars}
    time.sleep(0.3)

    yaw_rate = a.speed * math.tan(a.steer) / a.wheelbase     # the Twist the Ackermann system wants
    twist = Twist(); twist.linear.x = a.speed; twist.angular.z = yaw_rate
    step = WorldControl(); step.pause = True; step.multi_step = 10

    t_a = sim_ns.value; time.sleep(0.2)
    if sim_ns.value != t_a:
        print(f"the world is running on its own before the first step (clock {t_a * 1e-9:.3f} -> {sim_ns.value * 1e-9:.3f} s)")
        stop.value = 1; srv.kill(); return None
    target_ns = sim_ns.value
    waits, req_ms, retries = [], [], 0
    wall0 = time.monotonic()
    for t in range(a.ticks):
        for c in cars:
            pubs[c].publish(twist)
        target_ns += TICK_NS
        r0 = time.monotonic()
        for _attempt in range(6):
            ok, _ = node.request(f"/world/{WORLD}/control", step, WorldControl, Boolean, 500)
            if ok:
                break
            retries += 1
        else:
            print(f"tick {t}: the control service did not answer six times in a row"); stop.value = 1; srv.kill(); return None
        req_ms.append(1e3 * (time.monotonic() - r0))
        w0 = time.monotonic()
        while sim_ns.value < target_ns:                      # the request returns at once; the clock says when the steps ran
            if time.monotonic() - w0 > 5.0:
                print(f"tick {t}: the sim clock did not reach {target_ns * 1e-9:.3f} s in 5 s (at {sim_ns.value * 1e-9:.3f})")
                stop.value = 1; srv.kill(); return None
            time.sleep(0.0001)
        waits.append(time.monotonic() - w0)
    wall = time.monotonic() - wall0
    time.sleep(0.3)
    final = {c: (poses[3 * k], poses[3 * k + 1], poses[3 * k + 2]) for k, c in enumerate(cars)}
    sim_t = sim_ns.value * 1e-9
    n_lidar = lidar_count.value
    stop.value = 1; lis.join(timeout=3)
    srv.kill(); srv.wait()
    del pubs, node
    time.sleep(0.5)
    tr = np.array(traj[:]).reshape(-1, 2)
    tr = tr[~np.isnan(tr).any(axis=1)]                       # one stamped pose per tick, gaps dropped
    path = float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1))) if len(tr) > 1 else 0.0
    last = tr[-101:] if len(tr) > 101 else tr
    v_last = float(np.sum(np.linalg.norm(np.diff(last, axis=0), axis=1))) / (0.01 * max(1, len(last) - 1))
    res = dict(run=run, boot_s=boot, wall_s=wall, sim_s=a.ticks * 0.01, rtf=(a.ticks * 0.01) / wall, path_m=path, v_last=v_last,
               mean_tick_ms=1e3 * wall / a.ticks, mean_wait_ms=1e3 * float(np.mean(waits)), p99_wait_ms=1e3 * float(np.percentile(waits, 99)),
               mean_req_ms=float(np.mean(req_ms)), retries=retries, lidar_msgs=n_lidar, sim_t=sim_t, final=final)
    print(f"run {run}: server up in {boot:.1f} s; {a.ticks} ticks ({a.ticks * 0.01:.2f} sim s) in {wall:.2f} wall s -> "
          f"real-time factor {res['rtf']:.2f}; per tick {res['mean_tick_ms']:.2f} ms (request {res['mean_req_ms']:.2f} ms, "
          f"physics+sensors wait mean {res['mean_wait_ms']:.2f} ms, p99 {res['p99_wait_ms']:.2f} ms); requests retried {retries}; "
          f"lidar messages {n_lidar}; sim clock {sim_t:.3f} s")
    print(f"   the first car: {len(tr)} stamped poses, path {path:.3f} m in {a.ticks * 0.01:.2f} s; speed over the last "
          f"second {v_last:.2f} m/s (commanded {a.speed:.2f} m/s)")
    for c in cars:
        p = final[c]
        print(f"   {c}: x={p[0]:.6f} y={p[1]:.6f} z={p[2]:.6f}")
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
    print(f"speed: real-time factor {min(rtfs):.2f} to {max(rtfs):.2f} with {n_cars} car(s), each with a 1081-point GPU lidar at 40 Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
