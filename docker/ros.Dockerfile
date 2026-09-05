# M4: the ROS 2 image. One Jazzy container that runs BOTH the physics (ClearanceEnv on
# f1tenth_gym) and the rclpy nodes, so there is exactly one Python and one set of
# numeric libraries in the loop (ADR 0011, ADR 0012).
#
# The ROS release is a parameter. Moving to the next long-term release should be a
# one-line change here, tried on purpose, not found out by accident.
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base
ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO} \
    DEBIAN_FRONTEND=noninteractive \
    SDL_VIDEODRIVER=dummy \
    PIP_NO_CACHE_DIR=1

# System libraries: the same ones caatc-gym needs (opencv, pygame, shapely, numba),
# plus the ROS packages the demo uses. Every one of them is maintained for this release.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-venv git build-essential libgl1 libglib2.0-0 \
      ros-${ROS_DISTRO}-ackermann-msgs \
      ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-visualization-msgs \
      ros-${ROS_DISTRO}-rosbag2 \
 && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 does not let pip install into the system Python (PEP 668). A virtual
# environment that can still see the system packages keeps rclpy visible while our
# own dependencies live in the venv.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# The exact versions caatc-gym has, frozen from that image (docker/ros-constraints.txt),
# so both images compute with identical numeric libraries. The fingerprint check
# (M4 check 1) confirms it rather than assuming it.
COPY docker/ros-constraints.txt /tmp/ros-constraints.txt
WORKDIR /work
COPY pyproject.toml /work/
COPY caatc /work/caatc
RUN pip install -c /tmp/ros-constraints.txt .

# Our own messages, built once into the image. The Python nodes are run from the
# bind-mounted source during development (see run.sh).
COPY ros2/src/caatc_msgs /ws/src/caatc_msgs
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
 && cd /ws && colcon build --packages-select caatc_msgs \
 && rm -rf /ws/build /ws/log

COPY docker/ros-entrypoint.sh /ros-entrypoint.sh
ENTRYPOINT ["/ros-entrypoint.sh"]
CMD ["python3", "-m", "caatc.smoke"]
