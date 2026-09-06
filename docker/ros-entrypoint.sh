#!/bin/bash
# Source ROS 2 and our own message package, then run whatever was asked.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [ -f /ws/install/setup.bash ]; then
  source /ws/install/setup.bash
fi
exec "$@"
