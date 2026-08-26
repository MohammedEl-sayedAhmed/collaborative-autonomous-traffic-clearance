"""A fast subset of the headroom gate: naive << ideal, and monotone in yields."""
import numpy as np
import pytest

from caatc.scenario import easy_preset
from caatc.clearance_env import ClearanceEnv
from caatc.clearance_eval import run_episode
from caatc.baselines import NaiveHold, IdealCooperator


@pytest.fixture(scope="module")
def easy_env():
    env = ClearanceEnv(easy_preset())
    yield env
    env.close()


def test_ideal_beats_naive(easy_env):
    naive = [run_episode(easy_env, NaiveHold(), seed=s) for s in range(2)]
    ideal = [run_episode(easy_env, IdealCooperator(), seed=s) for s in range(2)]
    n_spd = np.mean([m["ev_mean_speed"] for m in naive])
    i_spd = np.mean([m["ev_mean_speed"] for m in ideal])

    assert all(not m["success"] for m in naive)       # naive truncates
    assert all(m["success"] for m in ideal)           # ideal clears
    assert i_spd / n_spd >= 3.0                        # real headroom (~4x by design)


def test_progress_monotone_in_yield_count(easy_env):
    progs = []
    for m in range(easy_env.cfg.num_cooperators + 1):
        recs = [run_episode(easy_env, IdealCooperator(yield_count=m), seed=s) for s in range(2)]
        progs.append(float(np.mean([r["ev_progress"] for r in recs])))
    # non-decreasing, and the optimum is meaningfully above the floor
    assert all(progs[i] >= progs[i - 1] - 0.02 for i in range(1, len(progs)))
    assert progs[-1] - progs[0] >= 0.2
