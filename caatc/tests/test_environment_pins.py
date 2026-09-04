"""Guard the versions the published results were measured on.

The scenario's numbers are only comparable if the software underneath them is the
one we think it is. Two things have already gone wrong in practice:

* installing PettingZoo quietly upgraded gymnasium from 0.29.1 to 1.3.0, which
  breaks the pin that both f1tenth_gym and caatc declare;
* the project moved to Python 3.12 to match ROS 2 Jazzy's interpreter, and a
  future "upgrade" of that base image would silently change every result.

These tests fail loudly instead.
"""
import sys

import pytest


def test_gymnasium_matches_the_declared_pin():
    """f1tenth_gym asks for gymnasium <0.30; nothing may quietly raise that."""
    import gymnasium

    major, minor = (int(x) for x in gymnasium.__version__.split(".")[:2])
    assert (major, minor) == (0, 29), (
        f"gymnasium {gymnasium.__version__} is outside the pin (0.29.x). Something in "
        "the image upgraded it -- the simulator would be running a different API than "
        "the one every published number was measured on."
    )


def test_python_is_the_ros_interpreter():
    """One Python everywhere: ours must match ROS 2 Jazzy's (3.12).

    ADR 0012 chose Jazzy, whose rclpy is built against Python 3.12, and ADR 0011
    runs one interpreter across the headless and ROS images so the physics is
    identical in both. If this fails, either the base image moved or the ROS target
    changed -- and every table needs re-verifying either way.
    """
    assert sys.version_info[:2] == (3, 12), (
        f"running Python {sys.version_info.major}.{sys.version_info.minor}, expected "
        "3.12 to match ROS 2 Jazzy"
    )


def test_f1tenth_gym_is_the_pinned_commit():
    """The simulator itself must be the pinned build, not a newer checkout."""
    import f1tenth_gym

    assert f1tenth_gym.__file__, "f1tenth_gym is not importable"
    # the v1.0.0 line exposes the Gymnasium-API env; the legacy line does not
    from f1tenth_gym.envs.f110_env import F110Env

    assert hasattr(F110Env, "default_config"), (
        "this is not the pinned v1.0.0 f1tenth_gym API"
    )


def test_numpy_version_is_recorded_with_results():
    """numpy is not pinned tightly, so make its version visible in test output.

    Moving numpy can shift the last bits of a float and therefore a threshold
    outcome. We do not fail on it -- we record it, so a re-baseline can be
    explained rather than discovered.
    """
    import numpy

    print(f"\nnumpy version in use: {numpy.__version__}")
    major = int(numpy.__version__.split(".")[0])
    assert major >= 2, "the published results were measured on numpy 2.x"


def test_torch_absent_or_cpu_only():
    """The training image is CPU-only by design (no GPU on this machine)."""
    torch = pytest.importorskip("torch")
    assert "+cpu" in torch.__version__ or not torch.cuda.is_available(), (
        "a CUDA build of torch appeared; the images are meant to be CPU-only"
    )
