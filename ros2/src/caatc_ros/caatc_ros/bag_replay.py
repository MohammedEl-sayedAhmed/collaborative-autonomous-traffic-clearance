"""Check 8: a car node fed ONLY its allow-listed topics from a bag must decide and drive exactly as it did live.

During a run the smoke records the allow-listed topics (``ros2 bag record``). Afterwards
this module replays, for each ROS car, only that car's topics into a fresh car node on a
private DDS domain, captures what the node publishes, and compares it with the bridge's
record of the live run: every decision (action and the 26 numbers it was taken from)
and every drive command (the float32 wire values) must be identical, tick for tick.

A node with any hidden input could not pass: the bag holds nothing else. And the check
keeps working after every simulator swap, including onto hardware, because it never
touches the simulator.

Usage (inside the caatc-ros image, on a domain of its own)::

    python3 -m caatc_ros.bag_replay --bag D/bag --records D --preset strict --policy numpy:... --v2v --cars 1,2,3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from caatc_msgs.msg import Decision
from rclpy.node import Node

from caatc.ros_bridge_core import Record, record_glob
from caatc.scenario import preset_config

from .msgs_io import stamp_tick_of

PLAY_DELAY_S = 3.0          # let the fresh node's subscriptions be discovered before the bag plays


def allowlisted_topics(cars: Sequence[int], num_agents: int, v2v: bool) -> List[str]:
    """The union of the ROS cars' allow-lists: what the recorder records, what the player plays."""
    topics = {"/caatc/episode"}
    for i in cars:
        topics |= {f"/car{i}/odom", f"/car{i}/joint_states"}
        if v2v:
            topics.add(f"/car{i}/v2v")
        else:
            topics |= {f"/car{k}/odom" for k in range(num_agents) if k != i}
    return sorted(topics)


def car_topics(car: int, num_agents: int, v2v: bool) -> List[str]:
    return allowlisted_topics([car], num_agents, v2v)


def record_command(bag_dir: str, topics: Sequence[str]) -> List[str]:
    return ["ros2", "bag", "record", "-o", bag_dir, "--storage", "mcap", *topics]


class Capture(Node):
    """Collects what one car node publishes, keyed by stamp tick (first copy wins)."""

    def __init__(self, car: int, sim_hz: float):
        super().__init__("bag_capture")
        self.sim_hz = sim_hz
        self.drives: Dict[int, Tuple[float, float]] = {}
        self.decisions: Dict[Tuple[int, int], dict] = {}
        # deep queues: the bag plays a whole episode in about a second
        self.create_subscription(AckermannDriveStamped, f"/car{car}/drive", self._on_drive, 5000)
        self.create_subscription(Decision, f"/car{car}/decision", self._on_decision, 1000)

    def _on_drive(self, msg: AckermannDriveStamped) -> None:
        st = stamp_tick_of(msg.header.stamp, self.sim_hz)
        self.drives.setdefault(st, (float(msg.drive.steering_angle), float(msg.drive.speed)))

    def _on_decision(self, msg: Decision) -> None:
        self.decisions.setdefault((int(msg.episode), int(msg.tick)),
                                  dict(action=int(msg.action), obs=np.asarray(msg.obs, dtype=np.float32)))


def replay_car(bag_dir: str, records: List[Record], preset: str, car: int, policy: str, v2v: bool,
               domain: int, log_dir: str) -> dict:
    """Play one car's topics from the bag into a fresh node; compare with the live records."""
    cfg = preset_config(preset)
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), PYTHONUNBUFFERED="1")
    topics = car_topics(car, cfg.num_agents, v2v)
    node_cmd = [sys.executable, "-m", "caatc_ros.car_node", "--preset", preset, "--car", str(car),
                "--policy", policy] + (["--v2v"] if v2v else [])
    play_cmd = ["ros2", "bag", "play", bag_dir, "--delay", str(PLAY_DELAY_S), "--topics", *topics]
    os.makedirs(log_dir, exist_ok=True)
    node_log = open(os.path.join(log_dir, f"replay_car{car}_node.log"), "wb")
    play_log = open(os.path.join(log_dir, f"replay_car{car}_play.log"), "wb")

    # the capture node lives in THIS process: rclpy reads ROS_DOMAIN_ID at init, so set it
    # here too, or the capture would listen on the default domain while the fresh node and
    # the player talk on the private one
    os.environ["ROS_DOMAIN_ID"] = str(domain)
    rclpy.init()
    cap = Capture(car, cfg.sim_hz)
    node = subprocess.Popen(node_cmd, stdout=node_log, stderr=subprocess.STDOUT, env=env)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            rclpy.spin_once(cap, timeout_sec=0.05)
        player = subprocess.Popen(play_cmd, stdout=play_log, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + 600.0
        while player.poll() is None and time.monotonic() < deadline:
            rclpy.spin_once(cap, timeout_sec=0.02)
        until = time.monotonic() + 1.5
        while time.monotonic() < until:
            rclpy.spin_once(cap, timeout_sec=0.02)
        player_code = player.returncode if player.poll() is not None else None
    finally:
        node.terminate()
        try:
            node.wait(timeout=10)
        except subprocess.TimeoutExpired:
            node.kill()
        cap.destroy_node()
        rclpy.try_shutdown()
        node_log.close(); play_log.close()

    # compare with the live records
    rep = dict(car=car, player_exit=player_code, ticks=0, drives_missing=0, drives_differ=0,
               decisions_total=0, decisions_missing=0, decisions_differ=0, node_exit=node.returncode)
    for rec in records:
        start = int(rec.meta["start_tick"]); ep = int(rec.meta["episode"])
        for t in range(len(rec.rows_applied)):
            rep["ticks"] += 1
            got = cap.drives.get(start + t)
            want = rec.wire[t][car - 1]
            if got is None:
                rep["drives_missing"] += 1
            elif np.float32(got[0]) != want[0] or np.float32(got[1]) != want[1]:
                rep["drives_differ"] += 1
        for b, t in enumerate(rec.boundary_ticks):
            rep["decisions_total"] += 1
            got = cap.decisions.get((ep, t))
            if got is None:
                rep["decisions_missing"] += 1
            elif got["action"] != int(rec.decisions[b][car - 1]) or not np.array_equal(got["obs"], rec.node_obs[b][car - 1]):
                rep["decisions_differ"] += 1
    rep["passed"] = (rep["ticks"] > 0 and rep["drives_missing"] == 0 and rep["drives_differ"] == 0
                     and rep["decisions_missing"] == 0 and rep["decisions_differ"] == 0)
    return rep


def replay_all(bag_dir: str, records_dir: str, preset: str, cars: Sequence[int], policy: str, v2v: bool,
               domain: int, log_dir: str) -> List[dict]:
    paths = sorted(glob.glob(os.path.join(records_dir, record_glob(preset))))
    records = sorted((Record.load(p) for p in paths), key=lambda r: int(r.meta["episode"]))
    return [replay_car(bag_dir, records, preset, car, policy, v2v, domain, log_dir) for car in cars]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Check 8: replay a bag's allow-listed topics into fresh car nodes.")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--records", required=True, help="directory with the bridge's .npz records of the same run")
    ap.add_argument("--preset", default="strict", choices=["easy", "hard", "strict"])
    ap.add_argument("--cars", default="1")
    ap.add_argument("--policy", default="local-ideal")
    ap.add_argument("--v2v", action="store_true")
    ap.add_argument("--domain", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the report here as JSON")
    a = ap.parse_args(argv)
    cars = [int(x) for x in a.cars.split(",") if x.strip()]
    domain = a.domain if a.domain is not None else 100 + os.getpid() % 100
    reps = replay_all(a.bag, a.records, a.preset, cars, a.policy, a.v2v, domain, a.records)
    for r in reps:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] car {r['car']}: {r['ticks']} ticks, drives missing {r['drives_missing']} "
              f"differ {r['drives_differ']}; {r['decisions_total']} decisions, missing {r['decisions_missing']} "
              f"differ {r['decisions_differ']} (player exit {r['player_exit']})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(reps, f, indent=1)
    return 0 if all(r["passed"] for r in reps) else 1


if __name__ == "__main__":
    sys.exit(main())
