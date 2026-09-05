# One cooperator's high-level decision at a step boundary, together with the
# observation it was taken from. Published by a car node, read by the bridge
# (for the reward bookkeeping) and by the checks (observation agreement).
std_msgs/Header header   # stamp = simulated time of the tick, tick * 0.01 s
uint32 tick              # physics tick the decision applies from (a multiple of cfg.substeps)
uint8 car                # cooperator index j, in the env's agent order (1..K)
uint8 action             # 0 STAY, 1 MERGE_LEFT, 2 MERGE_RIGHT, 3 SPEED_UP, 4 SLOW_DOWN
float32[] obs            # the 26 numbers the policy read, exactly as the node built them
