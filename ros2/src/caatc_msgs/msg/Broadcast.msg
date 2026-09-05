# What one car says over V2V: who it is, where it is, how fast it goes.
# Reserved for M4.2 (the relay); M4.1 uses the cars' Odometry topics directly.
uint8 car        # sender index in the env's agent order (0 = the emergency vehicle)
uint8 role       # 0 EV, 1 cooperator, 2 side-traffic occupant
float64 x
float64 y
float64 theta    # heading, radians
float64 v        # longitudinal speed, m/s
uint32 tick      # the tick the sender measured this at (delay shows up as an older tick)
