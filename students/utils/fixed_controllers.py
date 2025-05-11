import numpy as np

def alternating_gait(action_size, t):
    """Alternates actuation to mimic a walking gait."""
    action = np.zeros(action_size)
    
    # Alternate expansion & contraction every 10 timesteps
    if t % 20 < 10:
        action[:action_size // 2] = 1   # Front half expands
        action[action_size // 2:] = -1  # Back half contracts
    else:
        action[:action_size // 2] = -1  # Front half contracts
        action[action_size // 2:] = 1   # Back half expands
    
    return action



def sinusoidal_wave(action_size, t):
    """Generates a wave-like motion pattern for snake-like robots."""
    action = np.zeros(action_size)
    for i in range(action_size):
        action[i] = np.sin(2 * np.pi * (t / 20 + i / action_size))  # Sin wave pattern
    return action


def hopping_motion(action_size, t):
    """Makes the robot jump forward using periodic full-body contraction and expansion."""
    action = np.zeros(action_size)
    if t % 20 < 10:
        action[:] = 1  # Expand all active voxels
    else:
        action[:] = -1  # Contract all active voxels
    return action