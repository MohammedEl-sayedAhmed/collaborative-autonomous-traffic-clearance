# v1.0.0 line (ROS 2 / Python 3) — Python 3 dev image for the f1tenth_gym RL base.
# No ROS yet (M0/M1 are gym-only, per ADR 0005). ROS 2 Humble arrives in a later phase.
FROM python:3.11-slim

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
