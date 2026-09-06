"""The record-to-video tool draws every tick of a ROS record without ROS."""
import os

import pytest

from caatc.ros_bridge_core import Record, run_lockstep_inprocess
from caatc.scenario import strict_preset

cv2 = pytest.importorskip("cv2")


def test_a_record_renders_to_an_mp4(tmp_path):
    from caatc.ros_video import cars_at, render

    cfg = strict_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1, 2, 3], v2v=True)
    try:
        path = str(tmp_path / "ep.npz")
        bridge.record.save(path)
    finally:
        bridge.close()
    rec = Record.load(path)
    cars = cars_at(rec, cfg, 0)
    assert [c["role"] for c in cars] == ["ev", "coop", "coop", "coop"]
    out = render(rec, str(tmp_path / "ep.mp4"), fps=50, frame_skip=10)
    assert os.path.exists(out) and os.path.getsize(out) > 10_000
    cap = cv2.VideoCapture(out)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frames >= len(rec.rows_applied) // 10           # one frame per 10 ticks
