"""Stamps are the lockstep key, so tick <-> stamp must be exact in both directions."""
import pytest

from caatc.ros_tick import (EPOCH_TICKS, EPISODE_GAP_TICKS, episode_start_tick, ns_per_tick,
                            stamp_to_tick, tick_to_stamp)


def test_round_trip_is_the_identity_for_every_tick_a_long_run_can_reach():
    for t in range(0, 200_001):
        sec, ns = tick_to_stamp(t)
        assert 0 <= ns < 1_000_000_000
        assert stamp_to_tick(sec, ns) == t


def test_the_float_recipe_really_is_wrong_and_ours_is_not():
    """Documented failure of tick * 0.01: 410 -> 4.099999999 s."""
    bad = int(410 * 0.01 * 1e9)
    assert bad != 410 * 10_000_000            # the float recipe is off by one nanosecond
    assert tick_to_stamp(410) == (4, 100_000_000)


def test_a_stamp_off_a_tick_is_refused():
    with pytest.raises(ValueError):
        stamp_to_tick(4, 99_999_999)


def test_other_rates_and_bad_rates():
    assert ns_per_tick(50.0) == 20_000_000
    assert tick_to_stamp(3, sim_hz=50.0) == (0, 60_000_000)
    with pytest.raises(ValueError):
        ns_per_tick(30.0)                     # 1e9 / 30 is not a whole number of nanoseconds


def test_episodes_never_send_time_backwards():
    start = episode_start_tick(0, 0)
    assert start == EPOCH_TICKS > 0            # stamp 0 is never used
    end0 = start + 1400
    start1 = episode_start_tick(1, end0)
    assert start1 == end0 + EPISODE_GAP_TICKS > end0
