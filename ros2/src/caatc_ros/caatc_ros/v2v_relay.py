"""The V2V relay: the radio between the cars, as an rclpy shell around ``RelayCore``.

The relay is the air. It hears every car's odometry (the EV and the side traffic
included) and tells each cooperating car what it would hear over the radio: the
cars within range along the road, minus what a lossy channel dropped, as they were
``delay_ticks`` ago. From M4.2 on, this digest is the ONLY way a car node learns about
other cars; the raw odometry topics are no longer on its allow-list.

Every rule lives in ``caatc.ros_v2v.RelayCore`` (range, loss, delay, per-episode
seeding, the recorded drops, the cache for a re-published tick). This file only
turns messages into samples and digests back into messages.

Completeness works as in the car node: a tick is handled once every car's odometry
and the Episode message for that tick carry the same, newest stamp. A re-published
tick gets its cached digests again, once per re-publish (``EchoGate`` keyed on the
Episode message, which every re-publish carries exactly once).

Run inside the ``caatc-ros`` image::

    python3 -m caatc_ros.v2v_relay --preset strict [--range 25] [--loss 0.1] [--delay-ticks 2] [--seed 0]
                                   [--out-dir /src/saved_variables/ros]

At the end of every episode the relay writes ``relay-<preset>-seed<seed>-ep<n>.json`` with its
settings and every drop it made, so a lossy run can be replayed.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from caatc_msgs.msg import Broadcast, Episode, V2VDigest
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from caatc.ros_node_core import EchoGate, ProtocolError
from caatc.ros_v2v import RelayCore, required_range
from caatc.scenario import preset_config

from .msgs_io import sample_from_odometry, stamp_tick_or_breach

EPISODE_TOPIC = "/caatc/episode"
QOS_DEPTH = 10


class V2VRelay(Node):
    def __init__(self, preset: str, relay_range: Optional[float], loss: float, delay_ticks: int,
                 seed: int, out_dir: str):
        super().__init__("v2v_relay")
        self.cfg = preset_config(preset)
        self.core = RelayCore(self.cfg, relay_range, loss, delay_ticks, seed)
        self.out_dir = out_dir
        self.gate = EchoGate()
        self.latest: Dict[str, Tuple[int, object]] = {}
        self.episodes: Dict[int, Episode] = {}          # start_tick -> RUNNING Episode
        self.required: List[str] = [EPISODE_TOPIC] + [f"/car{k}/odom" for k in range(self.cfg.num_agents)]
        self._drops_written = 0

        self.create_subscription(Episode, EPISODE_TOPIC, self._on_episode, QOS_DEPTH)
        for k in range(self.cfg.num_agents):
            topic = f"/car{k}/odom"
            self.create_subscription(Odometry, topic, functools.partial(self._on_stamped, topic), QOS_DEPTH)
        self.pub = {i: self.create_publisher(V2VDigest, f"/car{i}/v2v", QOS_DEPTH)
                    for i in range(1, self.cfg.num_cooperators + 1)}

        log = self.get_logger()
        log.info(f"python {sys.executable}, numpy {np.__version__}")
        log.info(f"relay on '{preset}': range {self.core.range:g} m (required {required_range(self.cfg):g}), "
                 f"loss {self.core.loss:g}, delay {self.core.delay} ticks, seed {self.core.seed}; "
                 f"hears {self.cfg.num_agents} cars, serves {self.cfg.num_cooperators}")

    # -- incoming ------------------------------------------------------------------
    def _stamp_tick(self, topic: str, msg) -> int:
        return stamp_tick_or_breach(topic, msg.header.stamp, self.cfg.sim_hz)

    def _on_episode(self, msg: Episode) -> None:
        if msg.state == Episode.ENDED:
            self._write_record(msg)
            return
        if msg.preset.lower() != self.cfg.preset or msg.num_agents != self.cfg.num_agents:
            raise ProtocolError(f"the bridge runs '{msg.preset}' with {msg.num_agents} cars; "
                                f"this relay was started for '{self.cfg.preset}' with {self.cfg.num_agents}")
        self.episodes[int(msg.start_tick)] = msg
        stamp_tick = self._stamp_tick(EPISODE_TOPIC, msg)
        self.latest[EPISODE_TOPIC] = (stamp_tick, msg)
        self.gate.own_odom(stamp_tick)                   # the anchor message of a (re-)publish
        self._maybe_act()

    def _on_stamped(self, topic: str, msg) -> None:
        self.latest[topic] = (self._stamp_tick(topic, msg), msg)
        self._maybe_act()

    # -- the tick ------------------------------------------------------------------
    def _maybe_act(self) -> None:
        if any(t not in self.latest for t in self.required):
            return
        stamps = [self.latest[t][0] for t in self.required]
        newest = max(stamps)
        if not all(s == newest for s in stamps):
            return
        starts = [s for s in self.episodes if s <= newest]
        if not starts:
            return
        episode = self.episodes[max(starts)]
        tick = newest - int(episode.start_tick)
        samples = {k: sample_from_odometry(self.latest[f"/car{k}/odom"][1]) for k in range(self.cfg.num_agents)}
        fresh = (int(episode.episode), tick) not in self.core._cache
        digests = self.core.on_tick(int(episode.episode), tick, samples)
        if not self.gate.allow(newest, fresh):
            return
        stamp = self.latest[EPISODE_TOPIC][1].header.stamp
        for receiver, heard in digests.items():
            msg = V2VDigest()
            msg.header.stamp = stamp                     # echoed verbatim
            msg.header.frame_id = "map"
            msg.tick = int(tick)
            msg.receiver = int(receiver)
            msg.heard = [Broadcast(car=int(h.car), role=int(h.role), x=float(h.x), y=float(h.y),
                                   theta=float(h.theta), v=float(h.v), tick=int(h.tick)) for h in heard]
            self.pub[receiver].publish(msg)

    # -- the record -----------------------------------------------------------------
    def _write_record(self, ended: Episode) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        drops = [d for d in self.core.drops if d[0] == int(ended.episode)]
        path = os.path.join(self.out_dir, f"relay-{self.cfg.preset}-seed{int(ended.seed)}-ep{int(ended.episode)}.json")
        with open(path, "w") as f:
            json.dump(dict(preset=self.cfg.preset, seed=int(ended.seed), episode=int(ended.episode),
                           range=self.core.range, loss=self.core.loss, delay_ticks=self.core.delay,
                           relay_seed=self.core.seed, drops=drops), f)
        self.get_logger().info(f"episode {ended.episode} ended: {len(drops)} drops -> {path}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="The V2V relay: range, loss and delay between the cars.")
    ap.add_argument("--preset", default="strict", choices=["easy", "hard", "strict"])
    ap.add_argument("--range", type=float, default=None, help="metres along the road (default: the observation's largest gate)")
    ap.add_argument("--loss", type=float, default=0.0, help="probability a broadcast is dropped, per sender, receiver and tick")
    ap.add_argument("--delay-ticks", type=int, default=0, help="the digest carries positions this many ticks old")
    ap.add_argument("--seed", type=int, default=0, help="seed of the drop generator (re-seeded per episode)")
    ap.add_argument("--out-dir", default="/src/saved_variables/ros")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv if argv is None else argv
    a = parse_args(remove_ros_args(argv)[1:])
    rclpy.init(args=argv)
    node = V2VRelay(a.preset, a.range, a.loss, a.delay_ticks, a.seed, a.out_dir)
    code = 0
    try:
        rclpy.spin(node)
    except ProtocolError as e:
        node.get_logger().error(f"the lockstep contract was broken, stopping: {e}")
        code = 2
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
