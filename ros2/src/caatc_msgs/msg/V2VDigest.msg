# Everything one receiver heard this tick, after range, loss and delay were applied
# by the relay. Reserved for M4.2.
std_msgs/Header header
uint32 tick
uint8 receiver            # the car this digest is addressed to
caatc_msgs/Broadcast[] heard
