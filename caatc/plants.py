"""Plants by name: which physics sits behind the referee (M5, ADR 0013).

``"gym"`` is f1tenth_gym, the plant every published number was measured on (the referee's
default, so ``None``); ``"gazebo"`` is Gazebo Harmonic (needs the caatc-gazebo image). A record
carries the name in ``meta["plant"]`` so a replay uses the same kind of plant.
"""
from __future__ import annotations

from typing import Callable, Optional

from .scenario import ScenarioConfig

PLANTS = ("gym", "gazebo")


def make_plant(name: str, cfg: ScenarioConfig):
    """The plant object for ``name``; ``None`` means the referee's default (f1tenth_gym)."""
    if name == "gym":
        return None
    if name == "gazebo":
        from .gazebo_plant import GazeboPlant
        return GazeboPlant(cfg)
    raise ValueError(f"unknown plant {name!r}; one of {PLANTS}")


def plant_factory(name: str) -> Callable[[ScenarioConfig], Optional[object]]:
    if name not in PLANTS:
        raise ValueError(f"unknown plant {name!r}; one of {PLANTS}")
    return lambda cfg: make_plant(name, cfg)
