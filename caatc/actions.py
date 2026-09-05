"""The five things a cooperating car can decide to do.

Kept in a module with no other imports, so a car node (which has no simulator) can
import them without pulling in ``clearance_env`` or gymnasium. ``clearance_env``
re-exports them, so ``from caatc.clearance_env import STAY`` keeps working.
"""
STAY, MERGE_LEFT, MERGE_RIGHT, SPEED_UP, SLOW_DOWN = range(5)
ACTION_NAMES = ("STAY", "MERGE_LEFT", "MERGE_RIGHT", "SPEED_UP", "SLOW_DOWN")


def apply_decision(cfg, target_lane: int, target_speed: float, action: int):
    """Apply one high-level decision to a car's targets; return ``(lane, speed, changed)``.

    This is THE rule. ``ClearanceEnv.set_decision`` uses it for the simulator's
    bookkeeping and a ROS car node uses it for its own targets, so the two can never
    drift apart. ``changed`` is 1 when the target lane actually moved (the
    oscillation penalty and the ``lane_changes`` metric count those).
    """
    a = int(action)
    lane, speed, changed = int(target_lane), float(target_speed), 0
    if a == MERGE_LEFT:
        new = min(lane + 1, cfg.num_lanes - 1)
        changed = int(new != lane)
        lane = new
    elif a == MERGE_RIGHT:
        new = max(lane - 1, 0)
        changed = int(new != lane)
        lane = new
    elif a == SPEED_UP:
        speed = min(speed + cfg.coop_speed_delta, cfg.coop_speed_max)
    elif a == SLOW_DOWN:
        speed = max(speed - cfg.coop_speed_delta, cfg.coop_speed_min)
    elif a != STAY:
        raise ValueError(f"unknown decision {a} (expected 0..4)")
    return lane, speed, changed
