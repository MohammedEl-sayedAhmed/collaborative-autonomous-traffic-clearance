"""Check 7, subscription hygiene: who listens to what, sampled from the live ROS graph.

The car nodes are the deliverable, and the claim about them is that each hears only
its own state and its own V2V digest. That cannot be proven by reading their code
(a convenience subscription is the natural way to cheat), so this node watches the
graph itself while an episode runs:

* every 100 ms it asks the graph for each car node's subscriptions and unions them
  over the run; at the end the union must EQUAL the allow-list, no more and no less;
* ``/caatc/ground_truth`` is published on purpose, so the gate can demand that no node
  in a ``/car*`` namespace subscribes to it, at any sample;
* from M4.2 on, no ``/car{i}`` node may subscribe to another car's ``/car{k}/odom``;
* the set of nodes on the graph must equal the declared set, so no helper node can
  carry the extra subscription for a car.

The gate stops when the Episode message says ENDED for the last expected episode (or
after ``--max-seconds``), writes a JSON report, prints PASS/FAIL lines and exits 0 or 1.
It is one DDS participant looking at others, so it sees what the graph sees; a fleet
composed in one process would need the "K distinct process ids" criterion instead.

Run inside the ``caatc-ros`` image, on the same domain as the run::

    python3 -m caatc_ros.ros_gate --preset strict --episodes 2 --allowlist m4.2 --out /src/saved_variables/ros/gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from caatc_msgs.msg import Episode
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from caatc.scenario import preset_config

SAMPLE_PERIOD_S = 0.1
UNKNOWN = "_NODE_NAME_UNKNOWN_"


def allowlist(cfg, car: int, version: str) -> Set[Tuple[str, str]]:
    """The topics (name, type) car ``car`` may subscribe to, by contract version."""
    base = {("/caatc/episode", "caatc_msgs/msg/Episode"),
            (f"/car{car}/odom", "nav_msgs/msg/Odometry"),
            (f"/car{car}/joint_states", "sensor_msgs/msg/JointState")}
    if version == "m4.1":
        base |= {(f"/car{k}/odom", "nav_msgs/msg/Odometry") for k in range(cfg.num_agents) if k != car}
    elif version == "m4.2":
        base.add((f"/car{car}/v2v", "caatc_msgs/msg/V2VDigest"))
    else:
        raise ValueError(f"unknown allow-list version {version!r}")
    return base


class RosGate(Node):
    def __init__(self, preset: str, cars: List[int], version: str, episodes: int, expect_nodes: Set[Tuple[str, str]],
                 onboard: bool = False):
        super().__init__("ros_gate")
        self.onboard = onboard
        self.seen_iface: Dict[int, Set[Tuple[str, str]]] = {c: set() for c in cars}
        self.cfg = preset_config(preset)
        self.cars = cars
        self.version = version
        self.episodes_expected = episodes
        self.expect_nodes = expect_nodes
        self.seen_subs: Dict[int, Set[Tuple[str, str]]] = {c: set() for c in cars}
        self.ground_truth_listeners: Set[Tuple[str, str]] = set()
        self.foreign_odom: Set[Tuple[int, str]] = set()        # (car, topic it should not hear)
        self.unknown_seen = 0
        self.node_sets: List[Set[Tuple[str, str]]] = []
        self.samples = 0
        self.ended: Set[int] = set()
        self.cars_seen: Set[int] = set()
        self.create_subscription(Episode, "/caatc/episode", self._on_episode, 10)
        self.timer = self.create_timer(SAMPLE_PERIOD_S, self._sample)

    def _on_episode(self, msg: Episode) -> None:
        if msg.state == Episode.ENDED:
            self.ended.add(int(msg.episode))

    def _sample(self) -> None:
        self.samples += 1
        nodes = {(n, ns) for n, ns in self.get_node_names_and_namespaces() if n != "ros_gate"}
        self.node_sets.append(nodes)
        for car in self.cars:
            ns = f"/car{car}"
            if ("agent", ns) not in nodes:
                continue
            self.cars_seen.add(car)
            try:
                subs = self.get_subscriber_names_and_types_by_node("agent", ns)
            except Exception:                       # the node may have just vanished
                continue
            for topic, types in subs:
                for t in types:
                    self.seen_subs[car].add((topic, t))
                    if topic.startswith("/car") and topic.endswith("/odom") and topic != f"/car{car}/odom":
                        self.foreign_odom.add((car, topic))
        if self.onboard:
            for car in self.cars:
                ns = f"/car{car}"
                if ("vehicle_interface", ns) not in nodes:
                    continue
                try:
                    subs = self.get_subscriber_names_and_types_by_node("vehicle_interface", ns)
                except Exception:
                    continue
                for topic, types in subs:
                    for t in types:
                        self.seen_iface[car].add((topic, t))
                        if topic.startswith("/car") and not topic.startswith(f"/car{car}/"):
                            self.foreign_odom.add((car, topic))
        for info in self.get_subscriptions_info_by_topic("/caatc/ground_truth"):
            name, ns = info.node_name, info.node_namespace
            if name == UNKNOWN:
                self.unknown_seen += 1
                continue
            if ns.startswith("/car"):
                self.ground_truth_listeners.add((name, ns))

    def done(self) -> bool:
        return len(self.ended) >= self.episodes_expected and self.samples > 5

    def report(self) -> dict:
        per_car = {}
        ok_all = True
        for car in self.cars:
            want = allowlist(self.cfg, car, self.version)
            got = self.seen_subs[car]
            ok = got == want and car in self.cars_seen
            ok_all &= ok
            per_car[car] = dict(ok=ok, seen=car in self.cars_seen, extra=sorted(got - want), missing=sorted(want - got))
        per_iface = {}
        if self.onboard:
            from caatc_ros.onboard import interface_allowlist
            for car in self.cars:
                want = interface_allowlist(car); seen = self.seen_iface[car]
                per_iface[car] = dict(ok=seen == want, extra=sorted(seen - want), missing=sorted(want - seen), seen=sorted(seen))
        node_ok = bool(self.node_sets) and all(ns == self.expect_nodes for ns in self.node_sets[-3:])
        gt_ok = not self.ground_truth_listeners
        foreign_ok = not self.foreign_odom if self.version == "m4.2" else True
        iface_ok = all(v["ok"] for v in per_iface.values()) if per_iface else True
        return dict(passed=bool(ok_all and node_ok and gt_ok and foreign_ok and iface_ok), samples=self.samples,
                    per_car=per_car, per_interface=per_iface, node_set_ok=node_ok,
                    node_set_last=sorted(list(self.node_sets[-1])) if self.node_sets else [],
                    node_set_expected=sorted(list(self.expect_nodes)),
                    ground_truth_listeners=sorted(self.ground_truth_listeners),
                    foreign_odom=sorted(self.foreign_odom), unknown_samples=self.unknown_seen,
                    episodes_ended=sorted(self.ended))


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv if argv is None else argv
    ap = argparse.ArgumentParser(description="Check 7: subscription hygiene, sampled from the live graph.")
    ap.add_argument("--preset", default="strict", choices=["easy", "hard", "strict"])
    ap.add_argument("--cars", default="1", help="comma list of ROS-driven cars (agent indices)")
    ap.add_argument("--allowlist", default="m4.1", choices=["m4.1", "m4.2"])
    ap.add_argument("--episodes", type=int, default=1, help="stop after this many ENDED messages")
    ap.add_argument("--relay", action="store_true", help="a /v2v_relay node is expected on the graph")
    ap.add_argument("--expect-node", action="append", default=[],
                    help="another node expected on the graph, as name or name:namespace (repeatable), e.g. rosbag2_recorder")
    ap.add_argument("--onboard", action="store_true", help="M5.2: a vehicle_interface node per car is expected and audited")
    ap.add_argument("--max-seconds", type=float, default=900.0)
    ap.add_argument("--out", default="/src/saved_variables/ros/gate.json")
    a = ap.parse_args(remove_ros_args(argv)[1:])
    cars = [int(x) for x in a.cars.split(",") if x.strip()]
    expect = {("clearance_bridge", "/")} | {("agent", f"/car{c}") for c in cars}
    if a.relay:
        expect.add(("v2v_relay", "/"))
    for item in a.expect_node:
        name, _, ns = item.partition(":")
        expect.add((name, ns or "/"))

    rclpy.init(args=argv)
    if a.onboard:
        expect |= {("vehicle_interface", f"/car{c}") for c in cars} | {("ros_gz_bridge", "/")}
    gate = RosGate(a.preset, cars, a.allowlist, a.episodes, expect, onboard=a.onboard)
    t0 = time.monotonic()
    try:
        while rclpy.ok() and not gate.done() and time.monotonic() - t0 < a.max_seconds:
            rclpy.spin_once(gate, timeout_sec=0.05)
        rep = gate.report()
    except (KeyboardInterrupt, ExternalShutdownException):
        rep = gate.report()
    finally:
        gate.destroy_node()
        rclpy.try_shutdown()
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    for car, r in rep["per_car"].items():
        print(f"  [{'PASS' if r['ok'] else 'FAIL'}] car {car}: subscriptions equal the {a.allowlist} allow-list"
              + ("" if r["ok"] else f"  (extra {r['extra']}, missing {r['missing']}, seen {r['seen']})"))
    print(f"  [{'PASS' if not rep['ground_truth_listeners'] else 'FAIL'}] no car node listens to /caatc/ground_truth  {rep['ground_truth_listeners']}")
    for car, r in rep.get("per_interface", {}).items():
        print(f"  [{'PASS' if r['ok'] else 'FAIL'}] car {car}: the vehicle interface subscribes to its own sensors only"
              + ("" if r["ok"] else f"  extra {r['extra']} missing {r['missing']}"))
    print(f"  [{'PASS' if rep['node_set_ok'] else 'FAIL'}] the node set equals the declared set  (last {rep['node_set_last']})")
    if a.allowlist == "m4.2":
        print(f"  [{'PASS' if not rep['foreign_odom'] else 'FAIL'}] no car node hears another car's odometry  {rep['foreign_odom']}")
    print(f"  {rep['samples']} samples; episodes ended {rep['episodes_ended']} -> {a.out}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
