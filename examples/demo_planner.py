import numpy as np

from trailer_controller.path_reference import PathReference


def straight_path(length_m=20.0, ds=0.2, direction="reverse", start_pose=(0.0, 0.0, 0.0)):
    sign = -1.0 if str(direction).lower() == "reverse" else 1.0
    station = np.arange(0.0, length_m + ds, ds)
    x0, y0, yaw0 = start_pose
    tangent = np.array([np.cos(yaw0), np.sin(yaw0)])
    xy = np.array([x0, y0]) + np.outer(station, tangent)
    heading = yaw0 + (np.pi if sign < 0.0 else 0.0)
    return PathReference(
        x_r=xy[:, 0],
        y_r=xy[:, 1],
        theta_r=np.full_like(station, heading),
        s_r=station,
        dir_r=sign * np.ones_like(station),
    )


def arc_path(radius_m=15.0, angle_rad=np.pi / 4.0, ds=0.2, direction="reverse"):
    sign = -1.0 if str(direction).lower() == "reverse" else 1.0
    length = abs(radius_m * angle_rad)
    station = np.arange(0.0, length + ds, ds)
    phi = station / radius_m
    x = radius_m * np.sin(phi)
    y = radius_m * (1.0 - np.cos(phi))
    tangent = phi
    heading = tangent + (np.pi if sign < 0.0 else 0.0)
    return PathReference(x, y, heading, station, sign * np.ones_like(station))
