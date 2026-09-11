"""M5: plants by name."""
import pytest

from caatc.plants import PLANTS, make_plant, plant_factory
from caatc.scenario import easy_preset


def test_gym_is_the_default_and_unknown_names_are_refused():
    cfg = easy_preset()
    assert make_plant("gym", cfg) is None and plant_factory("gym")(cfg) is None
    assert "gazebo" in PLANTS
    with pytest.raises(ValueError):
        make_plant("unity", cfg)
    with pytest.raises(ValueError):
        plant_factory("unity")


def test_a_record_names_its_plant():
    from caatc.ros_bridge_core import BridgeCore
    cfg = easy_preset()
    b = BridgeCore(cfg, ros_cars=[1])
    try:
        b.begin_episode(0)
        assert b.record.meta["plant"] == "gym"
    finally:
        b.close()
