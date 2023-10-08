import math


def calculate_distance(x0, y0, z0, x1, y1, z1):
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)


def calculate_acceleration(d1, d2, delta_t):
    """
    Calculate average acceleration given two distances traveled over a fixed time interval.

    Parameters:
    - d1: Distance traveled in the first time interval.
    - d2: Distance traveled in the second time interval.
    - delta_t: Time interval (assumed to be the same for both distances).

    Returns:
    - Average acceleration over the time interval.
    """
    # Calculate average velocities
    v1 = d1 / delta_t
    v2 = d2 / delta_t

    # Calculate acceleration
    a = (v2 - v1) / delta_t

    return a
