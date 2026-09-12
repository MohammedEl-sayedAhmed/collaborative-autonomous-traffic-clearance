# M5: the 3D plant image. caatc-ros (Jazzy) plus Gazebo Harmonic, the Gazebo release
# that ROS 2 Jazzy pairs with officially, installed from the ROS apt repository only
# (ADR 0013). Harmonic is supported until May 2029, the same month as Jazzy.
#
# Headless: the GPU lidar renders with OGRE 2 through EGL, so the Mesa EGL and DRI
# libraries are here, but no display is needed. Nothing here has an end of life
# before May 2029.
ARG ROS_DISTRO=jazzy
FROM caatc-ros
ARG ROS_DISTRO
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-gz \
      libegl1 libgl1-mesa-dri libglx-mesa0 libegl-mesa0 \
 && rm -rf /var/lib/apt/lists/*
# M5.2: the car's own localization, from Nav2 for this release (lidar + a map -> a pose):
# the map server, AMCL, and the lifecycle manager that brings them up. Same support horizon.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-nav2-map-server \
      ros-${ROS_DISTRO}-nav2-amcl \
      ros-${ROS_DISTRO}-nav2-lifecycle-manager \
      ros-${ROS_DISTRO}-tf2-tools \
 && rm -rf /var/lib/apt/lists/*
# M5: the lockstep plant plugin (gazebo/plugins/lockstep): each tick's commands in and the state
# out through synchronous services, so a referee tick is exact and repeatable (no topic race).
# Built here against the Gazebo Harmonic that ROS ships. cmake and a compiler come with ros-base.
COPY gazebo/plugins/lockstep /tmp/lockstep
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
 && apt-get update && apt-get install -y --no-install-recommends cmake \
 && cmake -S /tmp/lockstep -B /tmp/lockstep/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/caatc \
 && cmake --build /tmp/lockstep/build -j4 && cmake --install /tmp/lockstep/build \
 && rm -rf /tmp/lockstep /var/lib/apt/lists/*
ENV GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/caatc/lib
# Gazebo looks for models and worlds here; run.sh mounts the repository at /src.
ENV GZ_SIM_RESOURCE_PATH=/src/gazebo/models:/src/gazebo/worlds
CMD ["gz", "sim", "--version"]
