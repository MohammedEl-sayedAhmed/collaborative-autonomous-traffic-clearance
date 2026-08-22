#!/bin/bash
set -e
source /opt/ros/kinetic/setup.bash
# Overlay the catkin workspace once it has been built.
if [ -f /ws/devel/setup.bash ]; then
  source /ws/devel/setup.bash
fi
exec "$@"
