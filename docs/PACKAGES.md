# Package & message reference

The workspace has **36 catkin packages**: 12 project packages, 16 vendored ROS navigation packages
(modified to add the communication layer), 5 hardware/system packages from the MIT racecar, and 2
message packages. Paths are under `simulator/racecar-simulator/` unless noted.

## Project packages

| Package | Role |
|---------|------|
| `racecar_gazebo` | Gazebo bring-up (`one/two/four_racecars.launch`); `gazebo_odometry.py` turns link states into 20 Hz odometry. |
| `racecar_description` | URDF/xacro for the racecar and the ambulance; meshes and world models. |
| `racecar_communication` | V2V: each car broadcasts ID/pose/lane/footprint; receivers range-gate and aggregate peers. |
| `racecar_navigation` | Per-car `move_base` config + carlike TEB params + the lane-decision scripts (`customLocalPlanner`, `threeLanes_current_future_pos`, `isGoalReached`, `move_base_follower`, `simpleGoal`). |
| `racecar_move_car` | `MoveCar` action layer: client + lane-keeping / velocity / lane-change action servers arbitrating nav vs. RL. |
| `racecar_control` | Stanley, OpenCV lane keeping, Krauss car-following, keyboard teleop, and the `servo_commands` output stage. |
| `racecar_localization` | AMCL launch config (map_server + amcl) for 1/2/4 cars. |
| `racecar_mapping` | gmapping launches + saved `threeLanes` / `threeLanesCurve` maps. |
| `racecar_rviz` | RViz view presets for navigation and mapping. |
| `racecar_clear_ev_route` | RL **agent**: `SAQLMaster` + `SingleAgentQlearning` tabular Q-learning. |
| `racecar_rl_environments` | RL **environment**: `ClearEVRouteBasicEnv` serving state/reward/lifecycle over ROS services. |
| `ackermann_msgs`, `vesc_msgs` | Ackermann drive + VESC telemetry message definitions. |

## Vendored ROS navigation stack (`navigation_/`)

`amcl`, `base_local_planner`, `carrot_planner`, `clear_costmap_recovery`, **`costmap_2d`**,
`dwa_local_planner`, `fake_localization`, `global_planner`, `map_server`, `move_base`,
`move_slow_and_clear`, `nav_core`, `navfn`, `rotate_recovery`, `voxel_grid` (+ the `navigation`
metapackage).

> This is the ROS `navigation` stack vendored into the workspace **and modified**: `costmap_2d`
> carries the project's custom `costmap_2d::CommunicationLayer` plugin
> (`plugins/communication_layer.cpp`, registered in `costmap_plugins.xml`). That's why the stack is
> in-tree rather than installed from apt.

## Hardware / system packages (`system/`)

`racecar` (bringup + `ackermann_cmd_mux`), `vesc` (`vesc_driver`, `vesc_ackermann`), `serial`,
`hokuyo_node`. These target the physical car. `ackermann_cmd_mux` **is used by the simulator** (the
priority mux for drive commands); `hokuyo_node` is a hardware lidar driver and is blacklisted from
the build.

## Custom messages, services & actions

### `racecar_communication`
| Type | Fields |
|------|--------|
| `ID.msg` | `string type`, `float32 x_position`, `float32 y_position`, `int8 lane_num`, `geometry_msgs/Polygon footprint`, `float32 velocity`, `float32 max_vel`, `float32 max_acc` |
| `IDStamped.msg` | `Header header`, `uint8 robot_num`, `ID id` |
| `IDsCombined.msg` | `Header header`, `uint8 robot_num`, `ID[6] ids` |
| `FootprintsCombined.msg` | `Header header`, `uint8 robot_num`, `geometry_msgs/Polygon[6] footprints` |

### `racecar_control`
| Type | Fields |
|------|--------|
| `drive_param.msg` | `float32 velocity`, `float32 angle` |

### `racecar_move_car`
| Type | Fields |
|------|--------|
| `MoveCar.action` | goal `Goal mcGoal` / result `int16 mcResult` / feedback `int16 mcFeedback` (−2 uninit, −1 in-progress, 0 infeasible, 1 done) |
| `Goal.msg` | `Header header`, `bool action_source` (0 nav / 1 RL), `int32 control_action` (0 keep / 1 left / 2 right), `float32 acc` |

### `racecar_navigation`
| Type | Fields |
|------|--------|
| `NavAction.msg` | `Header header`, `int16 control_action` (0 keep / 1 left / 2 right) |
| `Lanes_Info.msg` | `Header header`, `int16[] map_array` |
| `BoolStamped.msg` | `Header header`, `bool Bool` |
| `CustomLocalPlannerFeedback.srv` | req `bool laneChangeActive` → resp `bool adjusted` |

### `racecar_clear_ev_route` (RL agent)
| Type | Fields |
|------|--------|
| `RLPolicyActionService.srv` | req `int16 control_action`, `float32 acc`, `bool doneRLflag` → resp `bool RLActionresult`, `float32 RLActionTime` |

### `racecar_rl_environments` (RL environment)
| Type | Fields |
|------|--------|
| `states.srv` | req `int8 robot_num` → resp `float32 agent_vel`, `int8 agent_lane`, `float32 amb_vel`, `int8 amb_lane`, `float32 rel_amb_y` |
| `reward.srv` | req `float32 amb_last_velocity`, `float32 execution_time` → resp `float32 reward` |
| `startSim.srv` | req `int8 num_of_agents`, `int8 num_of_EVs` → resp `bool is_successful` |
| `resetORcloseSim.srv` | req `bool reset_close_sim` → resp `bool is_successful` |
| `areWeDone.msg` | `bool is_activated`, `int8 is_episode_done` (0 none, 1 max-time, 2 amb-goal, 3 agent-goal, 4 died/stuck) |

> `racecar_rl_environments` also ships legacy `areWeDone.srv` and a set of unused `*_server.py` /
> `*_client.py` scaffolds — the live environment node is `racecar_clear_ev_route_basic_env.py`.
