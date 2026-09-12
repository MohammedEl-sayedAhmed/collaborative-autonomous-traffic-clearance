"""M5.2: the map is the world's walls, rasterised."""
import os
import numpy as np

from caatc.frenet import CenterlineFrame
from caatc.gazebo_map import RESOLUTION, occupancy_grid, write_map
from caatc.gazebo_world import WALL_GAP_EVERY, WALL_MARGIN, wall_boxes, world_sdf
from caatc.scenario import centerline_xy, easy_preset, lane_center_d


def _cell(grid, origin, x, y, res=RESOLUTION):
    return grid[int((y - origin[1]) / res), int((x - origin[0]) / res)]


def test_walls_are_occupied_and_lanes_are_free():
    cfg = easy_preset()
    grid, origin = occupancy_grid(cfg)
    frame = CenterlineFrame(*centerline_xy(cfg))
    half = cfg.num_lanes * cfg.lane_width / 2.0 + WALL_MARGIN
    hits = 0
    for s in np.linspace(1.0, frame.length - 1.0, 40):
        for lane in range(cfg.num_lanes):
            x, y, _ = frame.frenet_to_xytheta(s, lane_center_d(cfg, lane))
            assert _cell(grid, origin, x, y) == 0                 # every lane centre is free
        for side in (1, -1):
            x, y, _ = frame.frenet_to_xytheta(s, side * half)
            hits += int(_cell(grid, origin, x, y) == 100)
    assert hits >= 60                                              # most wall samples land on a wall (some fall in notches)
    assert 0.5 < (grid == 100).mean() * 100 < 5.0                  # the walls are thin


def test_the_walls_have_notches_and_the_world_uses_the_same_boxes():
    cfg = easy_preset()
    boxes = wall_boxes(cfg)
    per_side = len(boxes) // 2
    n_seg = int(np.ceil(CenterlineFrame(*centerline_xy(cfg)).length / 0.5))
    assert per_side == n_seg - n_seg // WALL_GAP_EVERY              # every 8th segment is missing
    text = world_sdf(cfg, np.zeros((cfg.num_agents, 3)) + np.arange(cfg.num_agents)[:, None] * [2.0, 0, 0])
    assert text.count("<collision name=") - 1 == len(boxes)          # one collision per wall box, plus the ground
    assert "gz::sim::systems::Imu" in text


def test_write_map_produces_a_pgm_and_a_yaml(tmp_path):
    cfg = easy_preset()
    y = write_map(cfg, str(tmp_path))
    assert os.path.exists(y) and os.path.exists(str(tmp_path / "clearance.pgm"))
    txt = open(y).read()
    assert "resolution: 0.05" in txt and "origin: [-1.5000" in txt
    with open(str(tmp_path / "clearance.pgm"), "rb") as f:
        assert f.read(2) == b"P5"
