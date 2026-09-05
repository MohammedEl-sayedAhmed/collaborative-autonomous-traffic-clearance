"""Ticks and time stamps, in integers only.

The lockstep protocol keys every message on the tick it belongs to, carried as a
``builtin_interfaces/Time`` stamp. Building that stamp from ``tick * 0.01`` in
floating point is wrong for about 1.7% of ticks (rclpy truncates ``int(x * 1e9)``,
so 4.10 s becomes 4.099999999 s and a node decodes tick 409 instead of 410). So the
stamp is an exact integer function of the tick, and the inverse checks its input.

Simulated time keeps running across episodes (a bag or RViz must never see time go
backwards), so a *stamp tick* is ``episode_start + tick``; the bridge tells the
nodes ``episode_start`` in the Episode message. ``EPOCH_TICKS`` keeps stamp 0 unused:
tf2 and RViz treat time 0 as "unset".
"""
from __future__ import annotations

from typing import Tuple

NS_PER_S = 1_000_000_000
EPOCH_TICKS = 100          # the first episode starts at 1.000 s, not 0
EPISODE_GAP_TICKS = 100    # one simulated second between episodes


def ns_per_tick(sim_hz: float) -> int:
    """Nanoseconds per physics tick; must divide one second exactly."""
    hz = int(round(sim_hz))
    if hz <= 0 or abs(sim_hz - hz) > 1e-9 or NS_PER_S % hz:
        raise ValueError(f"sim_hz={sim_hz} does not divide one second into whole nanoseconds")
    return NS_PER_S // hz


def tick_to_stamp(stamp_tick: int, sim_hz: float = 100.0) -> Tuple[int, int]:
    """Absolute stamp tick -> ``(sec, nanosec)``, exactly."""
    if stamp_tick < 0:
        raise ValueError("ticks are never negative")
    total_ns = int(stamp_tick) * ns_per_tick(sim_hz)
    return total_ns // NS_PER_S, total_ns % NS_PER_S


def stamp_to_tick(sec: int, nanosec: int, sim_hz: float = 100.0) -> int:
    """``(sec, nanosec)`` -> absolute stamp tick; refuses a stamp that is not on a tick."""
    step = ns_per_tick(sim_hz)
    total_ns = int(sec) * NS_PER_S + int(nanosec)
    if total_ns % step:
        raise ValueError(f"stamp {sec}.{nanosec:09d} is not on a {sim_hz:g} Hz tick")
    return total_ns // step


def episode_start_tick(episode: int, previous_end_tick: int) -> int:
    """Where episode ``episode`` starts on the simulated clock.

    Episode 0 starts at ``EPOCH_TICKS``; each later one starts one simulated second
    after the previous one ended, so time is strictly increasing across the run.
    """
    if episode == 0:
        return EPOCH_TICKS
    return int(previous_end_tick) + EPISODE_GAP_TICKS
