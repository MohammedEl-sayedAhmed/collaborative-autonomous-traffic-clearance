"""A car node has no simulator. Its imports must not pull one in.

Run in a fresh interpreter, because pytest's own process may already have imported
``caatc.clearance_env`` for other tests.
"""
import subprocess
import sys

NODE_MODULES = ["caatc.actions", "caatc.scenario", "caatc.frenet", "caatc.controllers",
                "caatc.obs_spec", "caatc.decentralized", "caatc.ros_geometry", "caatc.ros_tick",
                "caatc.ros_node_core", "caatc.policy_export", "caatc.ros_v2v"]


def test_node_side_modules_do_not_import_the_simulator():
    code = (
        "import sys, importlib\n"
        f"for m in {NODE_MODULES!r}: importlib.import_module(m)\n"
        "bad = [m for m in ('caatc.clearance_env', 'gymnasium', 'f1tenth_gym', 'torch', 'stable_baselines3') if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "BAD:", out.stdout


def test_the_action_constants_are_the_same_object_everywhere():
    from caatc import actions, clearance_env, decentralized  # noqa: F401
    assert clearance_env.STAY is actions.STAY and clearance_env.SLOW_DOWN == 4
