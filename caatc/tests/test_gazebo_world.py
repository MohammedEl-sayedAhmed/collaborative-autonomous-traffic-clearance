"""M5.1: the Gazebo world is generated from the scenario (pure Python, no Gazebo)."""
import xml.dom.minidom as minidom

import numpy as np
import pytest

from caatc.gazebo_world import WORLD_NAME, car_name, world_sdf
from caatc.scenario import easy_preset, hard_preset


def test_the_world_holds_one_car_per_agent_in_agent_order_and_is_well_formed():
    cfg = hard_preset()
    poses = np.array([[1.0 * i, 0.1 * i, 0.01 * i] for i in range(cfg.num_agents)])
    text = world_sdf(cfg, poses)
    dom = minidom.parseString(text)
    names = [n.getElementsByTagName("name")[0].firstChild.data for n in dom.getElementsByTagName("include")]
    assert names == [car_name(i) for i in range(cfg.num_agents)]
    assert names[0] == "car0"                                  # the emergency vehicle is agent 0
    assert dom.getElementsByTagName("world")[0].getAttribute("name") == WORLD_NAME
    pose = dom.getElementsByTagName("include")[2].getElementsByTagName("pose")[0].firstChild.data.split()
    assert float(pose[0]) == pytest.approx(2.0) and float(pose[1]) == pytest.approx(0.2) and float(pose[5]) == pytest.approx(0.02)


def test_lane_lines_and_walls_follow_the_scenario():
    cfg = easy_preset()
    poses = np.zeros((cfg.num_agents, 3)); poses[:, 0] = np.arange(cfg.num_agents) * 2.0
    text = world_sdf(cfg, poses)
    assert text.count('<model name="lane_line_') == cfg.num_lanes + 1
    assert '<model name="wall_left">' in text and '<model name="wall_right">' in text
    bare = world_sdf(cfg, poses, lane_lines=False, walls=False)
    assert "lane_line_" not in bare and "wall_" not in bare
    with pytest.raises(ValueError):
        world_sdf(cfg, poses[:-1])
