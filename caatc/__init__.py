"""Collaborative Autonomous Traffic Clearance, the v1.0.0 line.

The ROS 2 / Python 3 rewrite (see docs/adr/). The learning core runs on f1tenth_gym
(Gymnasium API, several cars); the emergency-vehicle-clearing scenario sits on top.

This file imports nothing on purpose. A ROS 2 car node imports ``caatc.actions``,
``caatc.scenario``, ``caatc.frenet``, ``caatc.controllers`` and ``caatc.obs_spec`` and
must not pull in gymnasium or the simulator (``tests/test_node_imports.py``). The
Gymnasium registration of ``caatc/clearance-v0`` therefore lives in
``caatc.clearance_env``, and happens when that module is imported.
"""

__version__ = "1.0.0.dev0"
