"""The five things a cooperating car can decide to do.

Kept in a module with no other imports, so a car node (which has no simulator) can
import them without pulling in ``clearance_env`` or gymnasium. ``clearance_env``
re-exports them, so ``from caatc.clearance_env import STAY`` keeps working.
"""
STAY, MERGE_LEFT, MERGE_RIGHT, SPEED_UP, SLOW_DOWN = range(5)
ACTION_NAMES = ("STAY", "MERGE_LEFT", "MERGE_RIGHT", "SPEED_UP", "SLOW_DOWN")
