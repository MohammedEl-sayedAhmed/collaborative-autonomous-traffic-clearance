"""M5: the referee runs on any plant; the f1tenth_gym plant is the same physics as before."""
import numpy as np
import pytest

from caatc.clearance_env import ClearanceEnv
from caatc.plant import GymPlant, PlantState
from caatc.scenario import easy_preset


class FrozenPlant:
    """Cars that never move: proves the referee only talks to the plant through the interface."""

    def __init__(self, n):
        self.n, self.resets, self.steps = n, 0, 0
        self._state = None

    def reset(self, poses):
        self.resets += 1
        z = np.zeros(self.n)
        self._state = PlantState(x=poses[:, 0].copy(), y=poses[:, 1].copy(), theta=poses[:, 2].copy(), v=z.copy(), delta=z.copy())
        return self._state

    def substep(self, rows):
        self.steps += 1
        assert rows.shape == (self.n, 2)
        return self._state, False

    def collisions(self):
        return np.zeros(self.n)

    def close(self):
        pass


def test_the_referee_runs_a_whole_episode_on_a_plant_that_is_not_f1tenth():
    cfg = easy_preset()
    plant = FrozenPlant(cfg.num_agents)
    env = ClearanceEnv(cfg, plant=plant)
    assert env.inner is None                       # no f1tenth_gym object anywhere
    obs, info = env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(np.zeros(cfg.num_cooperators, dtype=int))
        steps += 1
        done = term or trunc
    assert trunc and not term                      # nobody moves: the clock runs out, no success, no crash
    assert not info["success"] and not info["collision"]
    assert plant.resets == 1 and plant.steps == steps * cfg.substeps
    with pytest.raises(RuntimeError):
        env._attach_renderer("rgb_array")          # rendering needs the f1tenth_gym plant
    env.close()


def test_the_gym_plant_reads_the_same_numbers_the_referee_records():
    cfg = easy_preset()
    env = ClearanceEnv(cfg)
    assert isinstance(env.plant, GymPlant) and env.inner is env.plant.inner
    env.reset(seed=0)
    st = env._last_state
    for i, c in enumerate(env.cars):
        assert (c["x"], c["y"], c["theta"], c["v"], c["delta"]) == (st.x[i], st.y[i], st.theta[i], st.v[i], st.delta[i])
    env.close()
