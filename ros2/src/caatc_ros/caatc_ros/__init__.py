"""caatc_ros: the ROS 2 shells of the Collaborative Autonomous Traffic Clearance demo.

Every module here is a thin shell around a pure core in the ``caatc`` package:
``clearance_bridge`` wraps ``caatc.ros_bridge_core.BridgeCore`` (the one
``ClearanceEnv``, driven one tick at a time), the car node wraps
``caatc.ros_node_core.NodeCore``. The shells only convert messages and move bytes;
the lockstep protocol, its bookkeeping and its record live in the cores, where
ordinary tests cover them without a message bus.

Start a node as ``python3 -m caatc_ros.<module>`` inside the ``caatc-ros`` image.
"""
