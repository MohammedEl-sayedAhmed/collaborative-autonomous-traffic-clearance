# Collaborative Autonomous Traffic Clearance — F1/10 graduation project (2020)
# ROS Kinetic + Gazebo 7 environment, fully containerized (nothing installed on host).

# --- Stage 1: fetch the Gazebo models the project worlds reference -----------
# The threeLanes* worlds use model:// URIs that Gazebo 7 would otherwise try to
# fetch from the (long-dead) online model database at runtime. Bake them in.
FROM ubuntu:24.04 AS gazebo_models
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN set -e; \
    for i in 1 2 3 4 5; do \
      git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/osrf/gazebo_models /gazebo_models && break \
      || { rm -rf /gazebo_models; echo "clone retry $i"; sleep 10; }; \
    done; \
    cd /gazebo_models; \
    git sparse-checkout set dumpster construction_barrel fire_hydrant jersey_barrier ground_plane sun; \
    rm -rf /gazebo_models/.git

# --- Stage 2: the actual ROS Kinetic environment ------------------------------
FROM osrf/ros:kinetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Dependencies beyond ros:kinetic-desktop-full, derived from the packages' real deps:
#   ros-control/ros-controllers/gazebo-ros-control/ackermann-msgs/joy  -> sim control stack
#   move-base-msgs   -> the vendored navigation_/ stack find_package()s it but does not vendor it
#   cmake-modules    -> build-only dep of six navigation_ packages, absent from desktop-full
#   tf2-geometry-msgs/tf2-sensor-msgs -> build deps of amcl/costmap_2d/dwa
#   teb-local-planner-> racecar_move_base.launch loads TebLocalPlannerROS params
#   gmapping         -> racecar_mapping SLAM demo
#   libsdl*/bullet/yaml-cpp/eigen -> map_server & friends hard-fail configure without them
#   python-jinja2    -> RL environment renders launch files from templates at runtime
#   xvfb/x11-utils/mesa-utils -> headless smoke-testing and GL debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-ros-control \
    ros-kinetic-ros-controllers \
    ros-kinetic-gazebo-ros-control \
    ros-kinetic-ackermann-msgs \
    ros-kinetic-joy \
    ros-kinetic-move-base-msgs \
    ros-kinetic-cmake-modules \
    ros-kinetic-tf2-geometry-msgs \
    ros-kinetic-tf2-sensor-msgs \
    ros-kinetic-teb-local-planner \
    ros-kinetic-gmapping \
    libsdl1.2-dev \
    libsdl-image1.2-dev \
    libbullet-dev \
    libyaml-cpp-dev \
    libeigen3-dev \
    python-jinja2 \
    xvfb \
    x11-utils \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gazebo_models /gazebo_models /opt/gazebo_models
# Empty DB URI stops Gazebo stalling on the dead online model database.
ENV GAZEBO_MODEL_PATH=/opt/gazebo_models \
    GAZEBO_MODEL_DATABASE_URI= \
    QT_X11_NO_MITSHM=1

RUN mkdir -p /ws/src
WORKDIR /ws

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
