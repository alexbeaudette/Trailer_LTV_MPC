import numpy as np


def wrap_to_pi(angle):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_difference(a, b):
    """Compute wrap_to_pi(a - b); subtract first, then wrap."""
    return wrap_to_pi(np.asarray(a) - np.asarray(b))


def clamp(value, low, high):
    return np.minimum(np.maximum(value, low), high)


def ramp_score(value, start_value, full_value):
    if full_value <= start_value:
        return float(value >= full_value)
    return float(clamp((value - start_value) / (full_value - start_value), 0.0, 1.0))
