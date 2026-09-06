# M4.3: the viewing side. caatc-ros plus RViz 2 and the X11 libraries a window needs.
# Only for looking; every check runs in caatc-ros, which has no GUI.
ARG ROS_DISTRO=jazzy
FROM caatc-ros
ARG ROS_DISTRO
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-rviz2 \
      libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0 libgl1-mesa-dri \
 && rm -rf /var/lib/apt/lists/*
ENV QT_X11_NO_MITSHM=1
CMD ["rviz2"]
