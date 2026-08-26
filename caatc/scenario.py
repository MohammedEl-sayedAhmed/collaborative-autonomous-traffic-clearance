"""Scenario configuration for the M1 ClearanceEnv.

Everything that defines the *problem* — road geometry, lanes, speeds, the start
layout, the ACC blocking law's knobs, the reward weights, and the EASY/HARD
preset — lives here as a plain dataclass so it is trivial to tweak, serialize,
and unit-test. Geometry (the world <-> frenet map) lives in ``frenet.py``; this
module only knows lanes as lateral offsets.

Design: ``docs/design/m1-clearance-env.md`` (ADR 0007). Blocking = **ACC on wide
lanes**; first trainable preset = **EASY**.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Tuple

import numpy as np

from .frenet import CenterlineFrame


@dataclass
class ScenarioConfig:
    """All knobs for one ClearanceEnv scenario.

    The defaults implement the EASY preset with the headroom worked out in the
    design doc: naive (nobody yields) clamps the EV to ``coop_speed`` so it needs
    ~``s_goal / coop_speed`` s to clear, while ideal (yield in time) lets the EV
    sprint at ``ev_max_speed`` (~``s_goal / ev_max_speed`` s) -> a ~4x ratio.
    ``max_time`` sits between the two so naive truncates and ideal succeeds.
    """

    # -- topology -------------------------------------------------------------
    num_cooperators: int = 3       # K trained cars (the EV is a +1 scripted car)
    num_lanes: int = 3
    lane_width: float = 0.9        # m (~3 car widths; wide, per ADR 0007)
    ev_lane: int = 1               # center lane index (0 .. num_lanes-1)

    # -- road (gentle sinusoid centerline; avoids degenerate zero curvature) --
    road_length: float = 60.0      # m of centerline x-extent
    road_amplitude: float = 0.5    # m lateral sinusoid amplitude (gentle)
    road_wavelength: float = 40.0  # m
    centerline_points: int = 600   # polyline resolution
    s_goal: float = 44.0           # EV goal arclength (m)

    # -- speeds ---------------------------------------------------------------
    ev_max_speed: float = 8.0      # m/s EV sprint target
    coop_speed: float = 2.0        # m/s cooperator cruise
    coop_speed_max: float = 4.0    # SPEED_UP cap
    coop_speed_min: float = 0.0    # SLOW_DOWN floor
    coop_speed_delta: float = 1.0  # SPEED_UP / SLOW_DOWN increment (m/s)

    # -- start layout (in frenet) --------------------------------------------
    ev_start_s: float = 0.0        # EV arclength at reset (center lane)
    coop_start_s: float = 8.0      # first cooperator arclength
    coop_gap: float = 10.0         # Delta stagger between cooperators (m)
    start_jitter: float = 0.5      # +/- m of per-seed jitter on each non-EV start s

    # -- EV adaptive-cruise (the blocking mechanism) -------------------------
    acc_standoff: float = 1.5      # d0: gap at which the EV target speed is 0 (m)
    acc_headway: float = 0.8       # tau: time headway (s); v_des = (gap-d0)/tau

    # -- timing / control -----------------------------------------------------
    sim_hz: float = 100.0          # physics rate (= 1 / f1tenth timestep 0.01)
    control_hz: float = 10.0       # high-level decision rate (frame-skip)
    max_time: float = 14.0         # T_max (s): between naive (~22s) and ideal (~6s)

    # -- reward weights (shared cooperative reward) --------------------------
    w_progress: float = 1.0        # per metre of EV forward progress (dense)
    w_ev_speed: float = 0.02       # per (m/s) EV speed per step (keep it moving)
    w_block: float = 0.05          # per step while the EV is blocked below sprint
    w_oscillation: float = 0.1     # per executed lane change (anti-chatter)
    w_collision: float = 100.0     # subtracted once on any car-car collision
    w_success: float = 50.0        # added once when the EV reaches the goal

    # -- observation ----------------------------------------------------------
    v2v_range: float = 25.0        # EV broadcast range gate (m)
    num_neighbors: int = 2         # M nearest cooperators included per obs
    clear_window: float = 6.0      # +/- s window for left_clear / right_clear (m)

    # -- vehicle (mirror f1tenth defaults; used only for gaps / geometry) -----
    car_length: float = 0.58
    car_width: float = 0.31

    # -- preset ---------------------------------------------------------------
    preset: str = "easy"           # "easy" | "hard"
    # HARD: for each cooperator, one adjacent side lane is blocked by a scripted
    # occupant so the free side must be read from V2V occupancy. Seeded per env.
    hard_block_sides: List[int] = field(default_factory=list)  # +1 left, -1 right (d is left-positive)

    seed: int = 12345

    # -- derived --------------------------------------------------------------
    @property
    def substeps(self) -> int:
        """Physics steps per high-level (wrapper) step."""
        return max(1, int(round(self.sim_hz / self.control_hz)))

    @property
    def dt(self) -> float:
        """Wrapper timestep (s) = substeps / sim_hz."""
        return self.substeps / self.sim_hz

    @property
    def max_steps(self) -> int:
        """T_max expressed in wrapper steps."""
        return int(np.ceil(self.max_time * self.control_hz))

    @property
    def num_occupants(self) -> int:
        return len(self.hard_block_sides) if self.preset == "hard" else 0

    @property
    def num_agents(self) -> int:
        """1 EV + K cooperators + H occupants (base-env agent count)."""
        return 1 + self.num_cooperators + self.num_occupants


# -- lane math (config-dependent; geometry-independent) -----------------------
def lane_center_d(cfg: ScenarioConfig, lane: int) -> float:
    """Lateral offset (m, left positive) of a lane's centerline."""
    return (lane - cfg.ev_lane) * cfg.lane_width


def lane_of(cfg: ScenarioConfig, d: float) -> int:
    """Nearest lane index for a lateral offset ``d`` (clamped to valid range)."""
    lane = int(round(d / cfg.lane_width)) + cfg.ev_lane
    return int(np.clip(lane, 0, cfg.num_lanes - 1))


# -- track construction -------------------------------------------------------
def centerline_xy(cfg: ScenarioConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Gentle-sinusoid centerline waypoints ``(xs, ys)``."""
    xs = np.linspace(0.0, cfg.road_length, cfg.centerline_points)
    ys = cfg.road_amplitude * np.sin(2.0 * np.pi * xs / cfg.road_wavelength)
    return xs, ys


def build_track(cfg: ScenarioConfig):
    """Build an f1tenth_gym ``Track`` from the scenario centerline.

    Imported lazily so ``scenario.py`` stays importable without f1tenth_gym
    (e.g. for the pure-math unit tests).
    """
    from f1tenth_gym.envs.track import Track

    xs, ys = centerline_xy(cfg)
    velx = np.full_like(xs, cfg.ev_max_speed)  # unused by us; required by API
    return Track.from_refline(x=xs, y=ys, velx=velx)


# -- start layout -------------------------------------------------------------
@dataclass
class Placement:
    """One car's start placement in frenet + its role."""
    s: float
    lane: int
    role: str  # "ev" | "coop" | "occupant"


def make_layout(cfg: ScenarioConfig, rng: np.random.Generator) -> List[Placement]:
    """Start layout, ordered as the base env's agents:
    ``[EV, coop_0..coop_{K-1}, occ_0..occ_{H-1}]``.

    EASY: cooperators are staggered in the EV (center) lane; side lanes empty.
    HARD: additionally one occupant per cooperator blocks a chosen adjacent side
    lane, so the free side must be inferred from V2V occupancy.

    ``rng`` perturbs only the **arclength** of each non-EV car by
    ``+/- start_jitter`` m, so ``reset(seed=...)`` yields genuinely different
    episodes while the **structure** (which agent is EV/coop/occupant, and every
    car's lane) stays seed-independent -- the env caches roles / occupant lanes
    once at construction and relies on that invariant. The EV is never jittered
    (fixed goal distance). Each occupant tracks its cooperator's jittered ``s``.
    """
    def jitter():
        return float(rng.uniform(-cfg.start_jitter, cfg.start_jitter)) if cfg.start_jitter else 0.0

    placements: List[Placement] = [Placement(cfg.ev_start_s, cfg.ev_lane, "ev")]

    coop_s = []
    for j in range(cfg.num_cooperators):
        s = cfg.coop_start_s + j * cfg.coop_gap + jitter()
        placements.append(Placement(s, cfg.ev_lane, "coop"))
        coop_s.append(s)

    if cfg.preset == "hard":
        for j, side in enumerate(cfg.hard_block_sides):
            s = coop_s[j] if j < len(coop_s) else cfg.coop_start_s + jitter()
            lane = int(np.clip(cfg.ev_lane + side, 0, cfg.num_lanes - 1))  # seed-independent
            placements.append(Placement(s, lane, "occupant"))

    return placements


# -- presets ------------------------------------------------------------------
def easy_preset(**overrides) -> ScenarioConfig:
    """EASY: side lanes empty; any merge direction clears the EV lane."""
    return replace(ScenarioConfig(preset="easy"), **overrides)


def hard_preset(**overrides) -> ScenarioConfig:
    """HARD: each cooperator has one adjacent side lane blocked (must read V2V).

    The blocked sides are seeded so a given ``seed`` is reproducible; the choice
    is materialized here (not at reset) so ``num_agents`` is stable.
    """
    base = ScenarioConfig(preset="hard")
    base = replace(base, **overrides)
    rng = np.random.default_rng(base.seed)
    # one blocked side (+1 left / -1 right) per cooperator, from valid sides
    sides = []
    for _ in range(base.num_cooperators):
        choices = []
        if base.ev_lane - 1 >= 0:
            choices.append(-1)
        if base.ev_lane + 1 <= base.num_lanes - 1:
            choices.append(+1)
        sides.append(int(rng.choice(choices)))
    return replace(base, hard_block_sides=sides)


EASY_PRESET = easy_preset()
HARD_PRESET = hard_preset()
