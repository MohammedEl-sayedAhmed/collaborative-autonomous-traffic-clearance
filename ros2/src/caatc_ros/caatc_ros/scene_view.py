"""The picture for RViz: the cars and the road as markers, plus a frame per car on /tf.

Nothing here is copied from any other project. The car is a box sized from the
simulator's own vehicle parameters (``ClearanceEnv.inner.params``: length, width), the
road is line strips sampled from ``scenario.centerline_xy`` (the same centre line the
physics and the observation use), the goal is a bar across the road. So no scenario
constant is written twice, and there is no file to keep in sync.

The node listens to every car's ``/car{i}/odom`` and to ``/caatc/episode`` (it is a
viewer, not a car node; the allow-list does not apply to it) and publishes:

* ``/caatc/scene`` (``visualization_msgs/MarkerArray``): the road, the lanes, the goal,
  and one box per car, coloured by role (EV red, cooperators blue turning green once
  they have left the EV's lane, side traffic grey);
* ``/tf``: ``map -> car{i}`` for every car, so RViz can follow one.

Run inside the ``caatc-ros`` image, on the run's DDS domain::

    python3 -m caatc_ros.scene_view --preset strict

RViz itself is not in the image (``ros:jazzy-ros-base`` has no GUI); a separate image
with ``ros-jazzy-rviz2`` and X11 is the viewing side. Without RViz, the markers are still
a checkable artifact: ``ros2 topic echo /caatc/scene`` shows them.
"""
from __future__ import annotations

import argparse
import functools
import sys
from typing import Dict, List, Optional

import numpy as np
import rclpy
from caatc_msgs.msg import Episode
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from caatc.frenet import CenterlineFrame
from caatc.ros_node_core import ROLE_COOPERATOR, ROLE_EV, role_of
from caatc.scenario import centerline_xy, lane_center_d, lane_of, preset_config

# the dashboard's colours, as RGBA
RGBA = {
    "ev": (0.851, 0.318, 0.369, 1.0),        # #d9515e
    "coop": (0.039, 0.612, 0.820, 1.0),      # #0a9cd1
    "coop_moved": (0.102, 0.682, 0.455, 1.0),  # #1aae74
    "occupant": (0.616, 0.624, 0.678, 1.0),  # #9d9fad
    "road_edge": (0.20, 0.21, 0.27, 1.0),
    "lane": (0.35, 0.36, 0.45, 1.0),
    "goal": (0.22, 0.678, 0.392, 1.0),       # #38ad64
}
CAR_LENGTH, CAR_WIDTH, CAR_HEIGHT = 0.58, 0.31, 0.12   # ClearanceEnv mirrors the simulator's defaults


def color(name: str) -> ColorRGBA:
    r, g, b, a = RGBA[name]
    return ColorRGBA(r=r, g=g, b=b, a=a)


class SceneView(Node):
    def __init__(self, preset: str, follow: Optional[int]):
        super().__init__("scene_view")
        self.cfg = preset_config(preset)
        self.frame = CenterlineFrame(*centerline_xy(self.cfg))
        self.follow = follow
        self.latest: Dict[int, Odometry] = {}
        self.pub = self.create_publisher(MarkerArray, "/caatc/scene", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Episode, "/caatc/episode", self._on_episode, 10)
        for i in range(self.cfg.num_agents):
            self.create_subscription(Odometry, f"/car{i}/odom", functools.partial(self._on_odom, i), 10)
        self.road = self._road_markers()
        self.get_logger().info(f"scene view on '{preset}': {self.cfg.num_agents} cars; markers on /caatc/scene, frames on /tf")

    # -- the static road, once ------------------------------------------------------
    def _road_markers(self) -> List[Marker]:
        cfg = self.cfg
        ss = np.linspace(0.0, self.frame.length, 240)
        markers = []
        for k in range(cfg.num_lanes + 1):
            d = (k - cfg.ev_lane - 0.5) * cfg.lane_width          # lane boundary offset
            m = Marker()
            m.header.frame_id = "map"
            m.ns, m.id, m.type, m.action = "road", k, Marker.LINE_STRIP, Marker.ADD
            edge = k in (0, cfg.num_lanes)
            m.scale.x = 0.05 if edge else 0.025
            m.color = color("road_edge" if edge else "lane")
            m.points = [Point(x=float(x), y=float(y), z=0.0)
                        for x, y, _ in (self.frame.frenet_to_xytheta(s, d) for s in ss)]
            markers.append(m)
        half = cfg.num_lanes * cfg.lane_width / 2.0
        g = Marker()
        g.header.frame_id = "map"
        g.ns, g.id, g.type, g.action = "goal", 0, Marker.LINE_STRIP, Marker.ADD
        g.scale.x = 0.08
        g.color = color("goal")
        g.points = [Point(x=float(x), y=float(y), z=0.0)
                    for x, y, _ in (self.frame.frenet_to_xytheta(cfg.s_goal, d) for d in (-half, half))]
        markers.append(g)
        return markers

    # -- the cars, every tick --------------------------------------------------------
    def _on_episode(self, msg: Episode) -> None:
        pass                                            # the view needs nothing from it yet

    def _on_odom(self, i: int, msg: Odometry) -> None:
        self.latest[i] = msg
        if i != 0:                                      # publish once per tick, on the EV's odometry
            return
        stamp = msg.header.stamp
        arr = MarkerArray()
        for m in self.road:
            m.header.stamp = stamp
            arr.markers.append(m)
        for car, od in sorted(self.latest.items()):
            box = Marker()
            box.header.stamp = stamp
            box.header.frame_id = "map"
            box.ns, box.id, box.type, box.action = "cars", car, Marker.CUBE, Marker.ADD
            box.pose = od.pose.pose
            box.pose.position.z = CAR_HEIGHT / 2.0
            box.scale.x, box.scale.y, box.scale.z = CAR_LENGTH, CAR_WIDTH, CAR_HEIGHT
            role = role_of(self.cfg, car)
            if role == ROLE_EV:
                box.color = color("ev")
            elif role == ROLE_COOPERATOR:
                _, d = self.frame.project(od.pose.pose.position.x, od.pose.pose.position.y)
                box.color = color("coop_moved" if lane_of(self.cfg, d) != self.cfg.ev_lane else "coop")
            else:
                box.color = color("occupant")
            arr.markers.append(box)
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = "map"
            tf.child_frame_id = f"car{car}"
            tf.transform.translation.x = od.pose.pose.position.x
            tf.transform.translation.y = od.pose.pose.position.y
            tf.transform.rotation = od.pose.pose.orientation
            self.tf.sendTransform(tf)
        self.pub.publish(arr)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv if argv is None else argv
    ap = argparse.ArgumentParser(description="Cars and road as RViz markers, frames on /tf.")
    ap.add_argument("--preset", default="strict", choices=["easy", "hard", "strict"])
    ap.add_argument("--follow", type=int, default=None, help="(reserved) the car RViz should follow")
    a = ap.parse_args(remove_ros_args(argv)[1:])
    rclpy.init(args=argv)
    node = SceneView(a.preset, a.follow)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
