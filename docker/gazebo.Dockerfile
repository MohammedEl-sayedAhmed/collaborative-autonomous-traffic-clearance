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
# Gazebo looks for models and worlds here; run.sh mounts the repository at /src.
ENV GZ_SIM_RESOURCE_PATH=/src/gazebo/models:/src/gazebo/worlds
CMD ["gz", "sim", "--version"]
