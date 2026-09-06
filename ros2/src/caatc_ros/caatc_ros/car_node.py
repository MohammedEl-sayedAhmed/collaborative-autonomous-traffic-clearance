"""The car node: one cooperating car as a ROS 2 node, a thin shell around ``NodeCore``.

The shell does three things and nothing else:

1. It listens to the bridge's state for one tick: ``/caatc/episode``, its own
   ``/car{i}/odom`` and ``/car{i}/joint_states``, and ``/car{k}/odom`` for every other
   car k (the EV included). That is the whole allow-list of M4.1; check 7 compares the
   node's subscriptions against it, so nothing else is subscribed here (no ``/clock``,
   no ``/caatc/ground_truth``).
2. When every one of those topics carries the same, newest stamp, the tick is complete.
   The stamp is an exact integer function of the tick (``caatc.ros_tick``), so the tick
   is ``stamp_tick - episode.start_tick`` and never a float. The Episode message is one
   of the stamped topics on purpose: the bridge publishes it with every tick's state,
   so waiting for "the Episode stamped with this tick" resolves the episode without a
   race at episode boundaries, and the final state after ``ENDED`` (which carries no
   RUNNING Episode) is never answered.
3. It hands the tick to ``NodeCore.on_tick`` and publishes what comes back: an
   ``AckermannDriveStamped`` on ``/car{i}/drive`` every tick and a ``Decision`` on
   ``/car{i}/decision`` at step boundaries. Both echo the own Odometry stamp verbatim.
   A repeated tick (the bridge re-published while waiting) gets the cached answer again,
   once per re-publish (``EchoGate``), not once per arriving message.

Everything about the protocol (skipped ticks, episode resets, the observation, the
policy, the low-level controller) lives in ``caatc.ros_node_core``. A ``ProtocolError``
from there stops the node with exit code 2.

Run inside the ``caatc-ros`` image, never through ``ros2 run``::

    python3 -m caatc_ros.car_node --preset strict --car 1

Imports are restricted: no ``caatc.clearance_env``, no gymnasium, no f1tenth_gym. The
preset comes from ``caatc.scenario.preset_config``, the same pure function the bridge
uses, so both sides build the identical configuration.
"""
from __future__ import annotations

import argparse
import functools
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from caatc_msgs.msg import Decision, Episode
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState

from caatc.ros_node_core import EchoGate, NodeCore, ProtocolError
from caatc.scenario import preset_config

from .msgs_io import sample_from_odometry, stamp_tick_or_breach

EPISODE_TOPIC = "/caatc/episode"
QOS_DEPTH = 10   # the default QoS profile: reliable, volatile, keep the last 10


class CarNode(Node):
    """Agent ``car`` (1..K) as node ``/car{car}/agent``."""

    def __init__(self, preset: str, car: int):
        super().__init__("agent", namespace=f"/car{car}")
        self.preset = preset.lower()
        self.cfg = preset_config(self.preset)
        self.car = int(car)
        self.core = NodeCore(self.cfg, self.car)      # rejects a car that is not a cooperator
        self.gate = EchoGate()                        # one answer per copy of the state
        self.own_odom = f"/car{self.car}/odom"
        self.own_joints = f"/car{self.car}/joint_states"

        # latest[topic] = (stamp tick, message) for every subscribed topic
        self.latest: Dict[str, Tuple[int, object]] = {}
        # the RUNNING Episode messages seen, by where their tick 0 sits on the clock
        self.episodes: Dict[int, Episode] = {}
        self.required: List[str] = [EPISODE_TOPIC, self.own_odom, self.own_joints] + [
            f"/car{k}/odom" for k in range(self.cfg.num_agents) if k != self.car]

        # the allow-list, and nothing else (use_sim_time stays false, so no /clock either)
        self.create_subscription(Episode, EPISODE_TOPIC, self._on_episode, QOS_DEPTH)
        for topic in self.required[1:]:
            kind = JointState if topic == self.own_joints else Odometry
            self.create_subscription(kind, topic, functools.partial(self._on_stamped, topic), QOS_DEPTH)
        self.pub_drive = self.create_publisher(AckermannDriveStamped, f"/car{self.car}/drive", QOS_DEPTH)
        self.pub_decision = self.create_publisher(Decision, f"/car{self.car}/decision", QOS_DEPTH)

        log = self.get_logger()
        log.info(f"python {sys.executable}, numpy {np.__version__}")
        log.info(f"car {self.car} on preset '{self.preset}': {self.cfg.num_agents} cars, "
                 f"{self.cfg.num_cooperators} cooperators, {self.cfg.substeps} ticks per decision")
        log.info("listening to: " + ", ".join(self.required))

    # -- incoming messages -----------------------------------------------------------
    def _stamp_tick(self, topic: str, msg) -> int:
        return stamp_tick_or_breach(topic, msg.header.stamp, self.cfg.sim_hz)

    def _on_episode(self, msg: Episode) -> None:
        if msg.state == Episode.ENDED:
            self.get_logger().info(f"episode {msg.episode} ended (seed {msg.seed})")
            return                                  # nothing to answer for a finished episode
        if msg.state != Episode.RUNNING:
            raise ProtocolError(f"Episode state {msg.state} is neither RUNNING nor ENDED")
        if (msg.preset.lower() != self.preset or msg.num_agents != self.cfg.num_agents
                or msg.num_cooperators != self.cfg.num_cooperators):
            raise ProtocolError(
                f"the bridge runs preset '{msg.preset}' with {msg.num_agents} cars and "
                f"{msg.num_cooperators} cooperators; this node was started for '{self.preset}' "
                f"with {self.cfg.num_agents} and {self.cfg.num_cooperators}")
        if int(msg.start_tick) not in self.episodes:
            self.get_logger().info(f"episode {msg.episode} running: seed {msg.seed}, "
                                   f"tick 0 at stamp tick {msg.start_tick}")
        self.episodes[int(msg.start_tick)] = msg
        self._on_stamped(EPISODE_TOPIC, msg)

    def _on_stamped(self, topic: str, msg) -> None:
        """Every callback ends here: note the stamp, then see whether the tick is complete."""
        stamp_tick = self._stamp_tick(topic, msg)
        self.latest[topic] = (stamp_tick, msg)
        if topic == self.own_odom:
            self.gate.own_odom(stamp_tick)
        self._maybe_act()

    # -- the tick ----------------------------------------------------------------------
    def _complete_stamp(self) -> Optional[int]:
        """The stamp tick every required topic carries, or None while the state is still arriving."""
        if any(topic not in self.latest for topic in self.required):
            return None
        stamps = [self.latest[topic][0] for topic in self.required]
        newest = max(stamps)
        return newest if all(s == newest for s in stamps) else None

    def _maybe_act(self) -> None:
        newest = self._complete_stamp()
        if newest is None:
            return
        # the episode whose tick 0 is the latest one at or before this stamp
        starts = [s for s in self.episodes if s <= newest]
        if not starts:
            return
        episode = self.episodes[max(starts)]
        tick = newest - int(episode.start_tick)

        samples = {k: sample_from_odometry(self.latest[f"/car{k}/odom"][1])
                   for k in range(self.cfg.num_agents)}
        own_odom = self.latest[self.own_odom][1]
        joints = self.latest[self.own_joints][1]
        if len(joints.position) < 1:
            raise ProtocolError(f"{self.own_joints} at tick {tick} carries no steering angle")

        out = self.core.on_tick(int(episode.episode), tick, samples, float(joints.position[0]))
        if not self.gate.allow(newest, out.fresh):
            return                                     # this copy of the state was answered already

        if out.decision is not None:
            dec = out.decision
            msg = Decision()
            msg.header.stamp = own_odom.header.stamp   # echoed verbatim
            msg.header.frame_id = f"car{self.car}"
            msg.episode = int(dec.episode)
            msg.tick = int(dec.tick)
            msg.car = int(dec.car)
            msg.action = int(dec.action)
            msg.obs = dec.obs.tolist()
            msg.s = float(dec.s)
            msg.d = float(dec.d)
            msg.lane = int(dec.lane)
            msg.tangent = float(dec.tangent)
            self.pub_decision.publish(msg)

        drive = AckermannDriveStamped()
        drive.header.stamp = own_odom.header.stamp     # the stamp of the tick this command is for
        drive.header.frame_id = f"car{self.car}"
        drive.drive.steering_angle = float(out.steer)  # float32 on the wire, as NodeCore rounded it
        drive.drive.speed = float(out.speed)
        self.pub_drive.publish(drive)

        if out.fresh and out.decision is not None:
            self.get_logger().debug(f"episode {episode.episode} tick {tick}: action {out.decision.action}, "
                                    f"steer {float(out.steer):+.4f} speed {float(out.speed):.3f}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="One cooperating car of the clearance demo, as a ROS 2 node.")
    ap.add_argument("--preset", default="strict", choices=["easy", "hard", "strict"],
                    help="the same preset the bridge runs (default: strict)")
    ap.add_argument("--car", type=int, default=1, help="agent index i, 1..K (default: 1)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv if argv is None else argv
    args = parse_args(remove_ros_args(argv)[1:])
    rclpy.init(args=argv)
    node = None
    code = 0
    try:
        node = CarNode(args.preset, args.car)
        rclpy.spin(node)                    # single-threaded: one callback at a time
    except ValueError as e:                 # a car index that is not a cooperator, and the like
        print(f"bad configuration: {e}", file=sys.stderr)
        code = 3
    except ProtocolError as e:
        node.get_logger().error(f"the lockstep contract was broken, stopping: {e}")
        code = 2
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
