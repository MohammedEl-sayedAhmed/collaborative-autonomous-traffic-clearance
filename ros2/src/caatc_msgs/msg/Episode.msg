# Run control: which episode is running, so nodes can reset between episodes and
# key every message on (episode, tick). Published by the bridge at every tick along
# with the state (so a late node cannot miss it), and once more with state = ENDED.
uint8 RUNNING = 0
uint8 ENDED = 1

std_msgs/Header header
uint32 episode           # 0, 1, 2, ... within one bridge process
uint8 state              # RUNNING or ENDED
int32 seed               # the reset seed of this episode
string preset            # easy | hard | strict; both sides build preset_config(preset) with defaults
uint32 start_tick        # absolute stamp tick of this episode's tick 0 (simulated time never runs backwards)
uint8 num_agents         # N: how many /car{i}/odom a node must wait for
uint8 num_cooperators    # K
