"""The fingerprint check must itself be trustworthy: exact here, and honest about a difference."""
import glob
import os

import numpy as np

from caatc import ros_fingerprint

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


def test_replay_is_identical_in_the_image_that_recorded_the_traces():
    paths = sorted(glob.glob(os.path.join(GOLDEN, "*.npz")))
    assert paths, "no golden traces"
    for p in paths:
        r = ros_fingerprint.replay(p)
        assert r["identical"], r


def test_a_perturbed_trace_is_reported_as_different(tmp_path):
    """Change one recorded reward by one unit in the last place: the verdict must flip."""
    src = sorted(glob.glob(os.path.join(GOLDEN, "easy-ideal-*.npz")))[0]
    g = dict(np.load(src))
    rewards = g["rewards"].copy()
    rewards[5] = np.nextafter(rewards[5], np.inf)
    g["rewards"] = rewards
    dst = tmp_path / "easy-ideal-tampered.npz"
    np.savez_compressed(dst, **g)
    r = ros_fingerprint.replay(str(dst))
    assert not r["identical"]
    assert r["first_diff_step"] == 5
    assert r["max_reward_diff"] > 0


def test_versions_report_the_pinned_simulator():
    v = ros_fingerprint.versions()
    assert v["f1tenth_gym"].startswith("5a301bd0"), v
    for k in ("python", "numpy", "numba", "gymnasium"):
        assert k in v and v[k]
