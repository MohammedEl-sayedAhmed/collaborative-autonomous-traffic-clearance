# One cooperator's high-level decision at a step boundary, with the observation it
# was taken from and the car's own view of where it is. Published by a car node,
# read by the bridge (reward bookkeeping) and by the checks (observation agreement,
# frame agreement).
std_msgs/Header header   # stamp = the Odometry stamp of the tick, echoed verbatim; informational
uint32 episode           # from the Episode message; the bridge keys on (episode, tick)
uint32 tick              # tick within the episode the decision applies from (a multiple of cfg.substeps)
uint8 car                # AGENT index i in the env's order: 1..K for cooperators (0 is the EV and never publishes)
                         # the env's cooperator index is j = car - 1
uint8 action             # 0 STAY, 1 MERGE_LEFT, 2 MERGE_RIGHT, 3 SPEED_UP, 4 SLOW_DOWN
float32[] obs            # the 26 numbers the policy read: np.clip(np.float32(per_coop_obs(...)), -10, 10)
float64 s                # the car's own road-frame values at this tick (for check 3)
float64 d
uint8 lane
float64 tangent          # the road's tangent angle at s
