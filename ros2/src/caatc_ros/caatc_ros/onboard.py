"""Onboard sensing (M5.2): what runs beside the car nodes when a car must sense for itself.

* the ``ros_gz_bridge`` configuration: each ROS car's lidar scan, IMU, joint states and its
  wheel-odometry TF from Gazebo into ROS, plus Gazebo's clock;
* the AMCL and map-server parameter files (M5.2, second step);
* the command lines for those processes, so ``ros_smoke`` only has to start them.

Pure functions, so they can be tested without ROS or Gazebo.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

from caatc.gazebo_world import WORLD_NAME

PARAMETER_BRIDGE = "/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge"


def bridge_config(cars: Sequence[int], world: str = WORLD_NAME) -> List[Dict[str, str]]:
    """One entry per bridged topic, Gazebo to ROS only."""
    rows: List[Dict[str, str]] = []
    for i in cars:
        rows += [
            dict(gz_topic_name=f"/world/{world}/model/car{i}/link/chassis/sensor/lidar/scan", ros_topic_name=f"/car{i}/scan",
                 gz_type_name="gz.msgs.LaserScan", ros_type_name="sensor_msgs/msg/LaserScan", direction="GZ_TO_ROS"),
            dict(gz_topic_name=f"/world/{world}/model/car{i}/link/chassis/sensor/imu/imu", ros_topic_name=f"/car{i}/imu",
                 gz_type_name="gz.msgs.IMU", ros_type_name="sensor_msgs/msg/Imu", direction="GZ_TO_ROS"),
            dict(gz_topic_name=f"/world/{world}/model/car{i}/joint_state", ros_topic_name=f"/car{i}/gz_joint_states",
                 gz_type_name="gz.msgs.Model", ros_type_name="sensor_msgs/msg/JointState", direction="GZ_TO_ROS"),
            dict(gz_topic_name=f"/model/car{i}/tf", ros_topic_name=f"/car{i}/tf",
                 gz_type_name="gz.msgs.Pose_V", ros_type_name="tf2_msgs/msg/TFMessage", direction="GZ_TO_ROS"),
        ]
    rows.append(dict(gz_topic_name=f"/world/{world}/clock", ros_topic_name="/clock",
                     gz_type_name="gz.msgs.Clock", ros_type_name="rosgraph_msgs/msg/Clock", direction="GZ_TO_ROS"))
    return rows


def write_bridge_config(path: str, cars: Sequence[int], world: str = WORLD_NAME) -> str:
    lines = []
    for r in bridge_config(cars, world):
        lines.append("- {" + ", ".join(f"{k}: {v}" for k, v in r.items()) + "}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def bridge_cmd(config_path: str) -> List[str]:
    return [PARAMETER_BRIDGE, "--ros-args", "-r", "__node:=ros_gz_bridge", "-p", f"config_file:={config_path}"]


def interface_cmd(python: str, preset: str, car: int) -> List[str]:
    """The vehicle interface, in the car's namespace, with its TF topics kept to that namespace."""
    return [python, "-m", "caatc_ros.vehicle_interface", "--preset", preset, "--car", str(car),
            "--ros-args", "-r", f"__ns:=/car{car}", "-r", f"/tf:=/car{car}/tf", "-r", f"/tf_static:=/car{car}/tf_static"]


def interface_allowlist(car: int):
    """What the vehicle interface may subscribe to: its own car's sensors and the operator's two topics."""
    return {("/caatc/episode", "caatc_msgs/msg/Episode"),
            (f"/car{car}/start_pose", "geometry_msgs/msg/PoseStamped"),
            (f"/car{car}/gz_joint_states", "sensor_msgs/msg/JointState"),
            (f"/car{car}/tf", "tf2_msgs/msg/TFMessage")}
