"""M4 check 1: is this image a numerical twin of the one the published numbers came from?

Prints the versions of everything that can move a floating-point result, replays the
golden traces in ``caatc/tests/golden/`` (recorded with the pre-seam env, never
regenerated) through a fresh ``ClearanceEnv``, and says one of two words:

* **IDENTICAL**: every observation, reward and flag of every trace matches exactly, so
  the M1-M3 tables carry over to this image as they are;
* **DIFFERENT**: something moved. The largest differences are printed, so they can be
  published rather than guessed at.

This is the one check that is allowed to report a difference instead of failing (see
the M4 design). It exits non-zero only when a trace cannot be replayed at all, or
when ``--gate`` is given and the M1 headroom gate fails in this image.

Usage (inside any image that has caatc installed and /src mounted)::

    python -m caatc.ros_fingerprint            # versions + golden replay
    python -m caatc.ros_fingerprint --gate     # ... and run the M1 headroom gate too
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
from typing import Dict, List

import numpy as np


def versions() -> Dict[str, str]:
    """Everything that can change the last digit of a result."""
    out: Dict[str, str] = {"python": platform.python_version(),
                           "platform": platform.platform()}
    for mod in ("numpy", "numba", "llvmlite", "scipy", "gymnasium", "shapely"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:  # pragma: no cover
            out[mod] = f"missing ({e.__class__.__name__})"
    out["f1tenth_gym"] = f1tenth_commit()
    try:
        import rclpy  # noqa: F401
        out["rclpy"] = os.environ.get("ROS_DISTRO", "present")
    except Exception:
        out["rclpy"] = "absent"
    return out


def f1tenth_commit() -> str:
    """The pinned commit, read from pip's record of the direct URL install."""
    try:
        from importlib import metadata

        dist = metadata.distribution("f1tenth_gym")
        raw = dist.read_text("direct_url.json")
        if raw:
            info = json.loads(raw)
            return info.get("vcs_info", {}).get("commit_id", "unknown")[:12]
    except Exception:
        pass
    return "unknown"


def replay(path: str) -> Dict:
    """Replay one golden trace; return exactness and the largest differences."""
    from caatc.clearance_env import ClearanceEnv
    from caatc.clearance_eval import preset_config

    name = os.path.basename(path).split("-")[0]
    g = np.load(path)
    env = ClearanceEnv(preset_config(name))
    result = {"trace": os.path.basename(path)[:-4], "steps": int(len(g["actions"])),
              "max_obs_diff": 0.0, "max_reward_diff": 0.0, "first_diff_step": None,
              "flags_match": True, "final_info_match": True}
    try:
        obs, _ = env.reset(seed=0)
        result["max_obs_diff"] = float(np.max(np.abs(obs - g["obs"][0])))
        te = tr = False
        info = {}
        for t, a in enumerate(g["actions"]):
            obs, r, te, tr, info = env.step(a)
            d_obs = float(np.max(np.abs(obs - g["obs"][t + 1])))
            d_r = abs(float(r) - float(g["rewards"][t]))
            if (d_obs > 0 or d_r > 0) and result["first_diff_step"] is None:
                result["first_diff_step"] = t
            result["max_obs_diff"] = max(result["max_obs_diff"], d_obs)
            result["max_reward_diff"] = max(result["max_reward_diff"], d_r)
            if te != bool(g["terminated"][t]) or tr != bool(g["truncated"][t]):
                result["flags_match"] = False
        final = json.loads(str(g["final_info"]))
        for k, v in final.items():
            got = info.get(k)
            if not ((got is None and v is None) or got == v):
                result["final_info_match"] = False
        result["ended_together"] = bool(te or tr)
    finally:
        env.close()
    result["identical"] = (result["max_obs_diff"] == 0.0 and result["max_reward_diff"] == 0.0
                           and result["flags_match"] and result["final_info_match"]
                           and result.get("ended_together", False))
    return result


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description="M4 check 1: numerical fingerprint of this image.")
    ap.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "tests", "golden"))
    ap.add_argument("--gate", action="store_true", help="also run the M1 headroom gate here")
    a = ap.parse_args(argv)

    print("== versions ==")
    for k, v in versions().items():
        print(f"  {k:<12} {v}")

    paths = sorted(glob.glob(os.path.join(a.golden, "*.npz")))
    if not paths:
        print(f"no golden traces under {a.golden}", file=sys.stderr)
        return 2
    print(f"\n== golden replay ({len(paths)} traces) ==")
    results = [replay(p) for p in paths]
    for r in results:
        mark = "identical" if r["identical"] else "DIFFERENT"
        extra = "" if r["identical"] else (f"  max|dobs|={r['max_obs_diff']:.3e} max|dr|={r['max_reward_diff']:.3e}"
                                          f" first_diff_step={r['first_diff_step']} flags={r['flags_match']}"
                                          f" final={r['final_info_match']}")
        print(f"  {r['trace']:<22} {r['steps']:>4} steps  {mark}{extra}")
    verdict = "IDENTICAL" if all(r["identical"] for r in results) else "DIFFERENT"
    print(f"\nVERDICT: {verdict}" + ("" if verdict == "IDENTICAL" else
          "  (the published tables do not carry over bit for bit; the differences above are the measurement)"))

    if a.gate:
        print("\n== M1 headroom gate, in this image ==")
        from caatc.clearance_smoke import main as gate_main

        code = gate_main([])
        if code:
            print("headroom gate FAILED in this image", file=sys.stderr)
            return int(code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
