# v1.0.0 line (ROS 2 / Python 3) — Python 3 dev image for the f1tenth_gym RL base.
#
# Pinned to Python **3.12** deliberately: ROS 2 Jazzy's rclpy is built against the
# distro interpreter (Ubuntu 24.04 / Python 3.12.3), and M4 runs ONE interpreter
# across the whole stack rather than two, so the physics is identical in the ROS
# image and the headless one (ADR 0011 fork 1, ADR 0012).
# Changing this line means re-verifying every published table -- do not "upgrade"
# it casually.
FROM python:3.12-slim

# System libs needed by f1tenth_gym's transitive deps (opencv, pygame, shapely,
# numba/llvmlite build). We run headless, so no display server is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV SDL_VIDEODRIVER=dummy \
    PIP_NO_CACHE_DIR=1

WORKDIR /work
# install deps first (pyproject pins f1tenth_gym @ the v1.0.0 commit) for layer caching
COPY pyproject.toml /work/
COPY caatc /work/caatc
RUN pip install .

CMD ["python", "-m", "caatc.smoke"]
