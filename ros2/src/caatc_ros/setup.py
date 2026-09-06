"""ament_python packaging for caatc_ros.

The nodes are always started as ``python3 -m caatc_ros.<module>`` with the image's
venv interpreter (``/opt/venv/bin/python3``), because a ``ros2 run`` console script
would carry a ``#!/usr/bin/python3`` shebang: a different numpy and no simulator
(see docs/design/m4-ros2-mechanical-demo.md, "Who runs the Python"). So no
console scripts are declared here on purpose; the package exists so colcon and
the ament index know about the nodes, their config and their dependencies.
"""
from setuptools import setup

package_name = "caatc_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        # the ament index marker, so `ros2 pkg list` finds the package
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mohammed El-sayed Ahmed",
    maintainer_email="57391064+MohammedEl-sayedAhmed@users.noreply.github.com",
    description=(
        "ROS 2 nodes for the Collaborative Autonomous Traffic Clearance demo: "
        "the clearance bridge (simulator and referee) and the car nodes."
    ),
    license="GPL-3.0-or-later",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)
