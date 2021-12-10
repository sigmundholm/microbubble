from os.path import join, split

import numpy as np
import matplotlib.pyplot as plt

from utils.plot import if_latex

base = split(split(split(split(split(__file__)[0])[0])[0])[0])[0]
bm_relative_path = "build/src/cutfem/navier_stokes/benchmarks"


def benchmark_data(level):
    pressure_folder = "pressure_q2_cn_lv1-6_dt4"
    pressure_path = join(base, bm_relative_path, pressure_folder, f"pointvalues_lv{level}")
    pressure_data = np.genfromtxt(pressure_path, delimiter=" ", skip_header=True)

    pressure_time = pressure_data[:, 1]
    pressure_a1 = pressure_data[:, 6]
    pressure_a2 = pressure_data[:, 11]
    pressure_diff = pressure_a1 - pressure_a2

    force_folder = "draglift_q2_cn_lv1-6_dt4"
    force_path = join(base, bm_relative_path, force_folder, f"bdforces_lv{level}")
    force_data = np.genfromtxt(force_path, delimiter=" ", skip_header=True)

    drag = force_data[:, 3]
    lift = force_data[:, 4]

    return pressure_time, drag, lift, pressure_diff


def benchmark_2d_3(level):
    full_path = join(base, bm_relative_path, "benchmark-2D-3.csv")
    head = list(map(str.strip, open(full_path).readline().split(",")))
    data = np.genfromtxt(full_path, delimiter=";", skip_header=True)
    time = data[:, 1]
    drag = data[:, 2]
    lift = data[:, 3]
    pressure = data[:, 4]

    bm_data = benchmark_data(level)
    bm_time, bm_drag, bm_lift, bm_pressure = bm_data

    # Drag
    if_latex(True)
    fig1, ax1 = plt.subplots()
    ax1.plot(bm_time, bm_drag, label=r"$\textrm{Benchmark}$")

    # TODO discard the very large values.
    drag[:9] = 0
    drag[-4:] = 0

    ax1.plot(time, drag, label=r"$C_D$")
    ax1.legend()
    ax1.set_title(r"$\textrm{Drag}$")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$C_D$")

    # Lift
    # TODO discard the very large values.
    lift[-2:] = 0
    fig2, ax2 = plt.subplots()
    ax2.plot(bm_time, bm_lift, label=r"$\textrm{Benchmark}$")
    ax2.plot(time, lift, label="$C_L$")
    ax2.legend()
    ax2.set_title(r"$\textrm{Lift}$")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("$C_L$")

    # Pressure
    fig3, ax3 = plt.subplots()
    ax3.plot(bm_time, bm_pressure, label=r"$\textrm{Benchmark}$")
    ax3.plot(time, pressure, label="$\Delta p$")
    ax3.legend()
    ax3.set_title(r"$\textrm{Pressure}$")
    ax3.set_xlabel("$t$")
    ax3.set_ylabel("$\Delta p$")

    plt.show()


if __name__ == '__main__':
    # TODO note: some values are discarded
    plt.show()
    benchmark_2d_3(6)
