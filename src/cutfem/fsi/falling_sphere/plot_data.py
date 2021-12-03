from os.path import join, split

import numpy as np
import matplotlib.pyplot as plt

from utils.plot import if_latex

base = split(split(split(split(split(__file__)[0])[0])[0])[0])[0]
relative_path = "build/src/cutfem/fsi/falling_sphere"


# if_latex(True)

def plot_positions():
    path = join(base, relative_path, "fsi_falling_sphere.csv")
    data = np.genfromtxt(path, delimiter=";", skip_header=True)

    time = data[:, 1]
    pos_x = data[:, 2]
    pos_y = data[:, 3]
    plot_point_path(pos_x, pos_y, "Position")
    v_x = data[:, 4]
    v_y = data[:, 5]
    plot_point_path(v_x, v_y, r"Velocity")
    a_x = data[:, 6]
    a_y = data[:, 7]
    plot_point_path(a_x, a_y, r"Acceleration")

    angle = data[:, 8]
    angular_velocity = data[:, 9]
    angular_acceleration = data[:, 10]
    plot_value_against_time(time, angle, "Angle")
    plot_value_against_time(time, angular_velocity, "Angular velocity")
    plot_value_against_time(time, angular_acceleration, "Angular acceleration")


def plot_forces(filename, name_prefix, checks=False, max_idx=None):
    path = join(base, relative_path, filename)
    data = np.genfromtxt(path, delimiter=";", skip_header=True)

    ks = data[:, 0]
    time = data[:, 1]

    a_gx = data[:max_idx, 2]
    a_gy = data[:max_idx, 3]
    a_sx = data[:max_idx, 4]
    a_sy = data[:max_idx, 5]
    a_x = data[:max_idx, 6]
    a_y = data[:max_idx, 7]

    plot_point_path(a_gx, a_gy, f"{name_prefix}gravity")
    plot_point_path(a_sx, a_sy, f"{name_prefix}surface force acc")
    plot_point_path(a_x, a_y, f"{name_prefix}total acc")

    if checks:
        # Check
        fig, ax = plt.subplots()
        ax.plot(ks, a_gx + a_sx - a_x, label="diff a_x")
        ax.plot(ks, a_gy + a_sy - a_y, label="diff a_y")
        ax.set_title(f"{name_prefix} diff plot")
        ax.legend()

        fig2, ax2 = plt.subplots()
        ax2.plot(ks, a_gx, + a_sx, label="a_x summed")
        ax2.plot(ks, a_gy, + a_sy, label="a_y summed")
        ax2.set_title(f"{name_prefix} summed acceleration")
        ax2.legend()


def plot_point_path(pos_x, pos_y, title=""):
    fig, ax = plt.subplots()
    # ax.axis('equal')

    ax.plot(pos_x[0], pos_y[0], "ro", color="green")
    for i in range(1, len(pos_x)):
        ax.plot([pos_x[i - 1], pos_x[i]], [pos_y[i - 1], pos_y[i]], color="gray", marker=".")

    ax.set_title(title)


def plot_value_against_time(time, angle, title=""):
    fig, ax = plt.subplots()
    ax.plot(time, angle)
    ax.set_title(title)


if __name__ == '__main__':
    # plot_positions()
    # name = "fsi_falling_sphere_forces-nosurf.csv"
    # plot_forces(name, "No surface forces: ")
    surf = "fsi_falling_sphere_forces.csv"
    plot_forces(surf, "With surface forces: ")

    plt.show()
