"""The scene renderer and the env's frame plumbing.

These exist because the renderer is easy to break silently: an earlier fix stored
the CenterlineFrame as ``self.frame``, shadowing the ``frame()`` drawing method, and
the default renderer crashed with "'CenterlineFrame' object is not callable" -- while
every other test stayed green.
"""
import numpy as np
import pytest

from caatc.render2d import BGR, SceneRenderer
from caatc.scenario import easy_preset, hard_preset, lane_center_d
from caatc.clearance_env import ClearanceEnv


def _cars(cfg, frame):
    """A minimal car list in the shape ClearanceEnv._cars() produces."""
    mk = lambda s, lane, role, v=2.0: dict(
        i=0, role=role, x=0.0, y=0.0, theta=frame.tangent_angle(s), v=v, delta=0.0,
        s=s, d=lane_center_d(cfg, lane), lane=lane)
    return [mk(4.0, cfg.ev_lane, "ev", 7.0),
            mk(9.0, cfg.ev_lane, "coop"),
            mk(19.0, cfg.num_lanes - 1, "coop")]


@pytest.fixture(scope="module")
def env():
    e = ClearanceEnv(easy_preset(), render_mode="rgb_array")
    yield e
    e.close()


def test_frame_is_a_drawable_method_and_returns_an_image():
    cfg = easy_preset()
    e = ClearanceEnv(cfg)          # no render_mode: we only need its frenet frame
    try:
        r = SceneRenderer(cfg, frame=e.frame)
        assert callable(r.frame), "frame() must stay a method (nothing may shadow it)"
        img = r.frame(_cars(cfg, e.frame), {"sim_time": 1.0, "ev_v": 7.0,
                                            "ev_blocked": False, "ev_progress": 0.1,
                                            "lane_changes": 0})
        assert img.shape == (r.H, r.W, 3) and img.dtype == np.uint8
        assert img.any(), "a blank frame means nothing was drawn"
    finally:
        e.close()


def test_ev_is_drawn_where_the_transform_says_it_is():
    cfg = easy_preset()
    e = ClearanceEnv(cfg)
    try:
        r = SceneRenderer(cfg, frame=e.frame)
        cars = _cars(cfg, e.frame)
        img = r.frame(cars, {"sim_time": 0.0, "ev_v": 7.0, "ev_blocked": False,
                             "ev_progress": 0.0, "lane_changes": 0})
        ev = cars[0]
        s0 = max(0.0, ev["s"] - 6.0)
        px, py = r._x(ev["s"], s0), r._y(ev["d"])
        assert tuple(int(c) for c in img[py, px]) == BGR["ev"], "EV not at its own transform"
        # a cooperator still in the EV lane is drawn in the "not yet moved" colour
        c1 = cars[1]
        assert tuple(int(c) for c in img[r._y(c1["d"]), r._x(c1["s"], s0)]) == BGR["coop"]
    finally:
        e.close()


def test_blocked_and_clear_frames_differ():
    cfg = easy_preset()
    e = ClearanceEnv(cfg)
    try:
        r = SceneRenderer(cfg, frame=e.frame)
        cars = _cars(cfg, e.frame)
        base = dict(sim_time=1.0, ev_v=2.0, ev_progress=0.1, lane_changes=0)
        blocked = r.frame(cars, {**base, "ev_blocked": True})
        clear = r.frame(cars, {**base, "ev_blocked": False})
        assert not np.array_equal(blocked, clear), "the HUD must show the ACC state"
    finally:
        e.close()


def test_hard_preset_with_occupants_renders():
    cfg = hard_preset()
    e = ClearanceEnv(cfg)
    try:
        r = SceneRenderer(cfg, frame=e.frame)
        e.reset(seed=0)
        cars = e._cars(e._last_obs)
        assert any(c["role"] == "occupant" for c in cars)
        img = r.frame(cars, {"sim_time": 0.0, "ev_v": 0.0, "ev_blocked": True,
                             "ev_progress": 0.0, "lane_changes": 0})
        assert img.shape == (r.H, r.W, 3)
    finally:
        e.close()


def test_rgb_array_frames_are_buffered_and_drained(env):
    env.reset(seed=0)
    env.pop_frames()                                  # discard the reset frame
    env.step(np.zeros(env.cfg.num_cooperators, dtype=int))
    frames = env.pop_frames()
    assert len(frames) == env.cfg.substeps, "one frame per physics substep"
    assert frames[0].ndim == 3 and frames[0].shape[2] == 3
    assert env.pop_frames() == [], "draining must clear the buffer"


def test_frame_buffer_is_bounded(env):
    env.reset(seed=0)
    cap = env._frames.maxlen
    assert cap is not None and cap > 0, "an unbounded buffer would grow to GBs"
    for _ in range(6):
        env.step(np.zeros(env.cfg.num_cooperators, dtype=int))
    assert len(env._frames) <= cap
