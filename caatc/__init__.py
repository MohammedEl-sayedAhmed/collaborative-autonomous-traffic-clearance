"""Collaborative Autonomous Traffic Clearance — v1.0.0 line.

The ROS 2 / Python 3 rewrite (see docs/adr/). The RL core is built on
f1tenth_gym (Gymnasium, N-agent); the cooperative emergency-vehicle-clearing
layer (EV designation, V2V, cooperative reward) lands on top in later milestones.
"""

__version__ = "1.0.0.dev0"


def _register_envs() -> None:
    """Register ``caatc/clearance-v0`` with Gymnasium (idempotent).

    The entry point is a string, so f1tenth_gym is not imported here -- only when
    the env is actually constructed (``gym.make`` / ``ClearanceEnv(...)``).
    """
    try:
        import gymnasium as gym

        if "caatc/clearance-v0" not in gym.registry:
            gym.register(
                id="caatc/clearance-v0",
                entry_point="caatc.clearance_env:ClearanceEnv",
            )
    except Exception:
        # gymnasium unavailable (e.g. pure-math test context) -- non-fatal.
        pass


_register_envs()
