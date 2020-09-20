import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import numpy as np

import os

# Latex font
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Remove small tick lines on the axes, that doesnt have any number with them.
matplotlib.rcParams['xtick.minor.size'] = 0
matplotlib.rcParams['xtick.minor.width'] = 0
matplotlib.rcParams['ytick.minor.size'] = 0
matplotlib.rcParams['ytick.minor.width'] = 0


def convergence_plot(ns, errors, yscale="log2", desired_order=2, reference_line_offset=0.5,
                     xlabel="$N$", title="", ax=None):
    res = np.polyfit(np.log(ns), np.log(errors), deg=1)
    print("Polyfit:", res)
    print("Order of convergence", -res[0])

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(ns, errors, "-o")
    if yscale == "log2":
        ax.set_yscale("log", basey=2)
    else:
        ax.set_yscale("log")

    ax.set_xscale("log")
    ax.set_title(
        r"\textrm{" + title + r"} \small{\textrm{Convergence order: " + str(-round(res[0], 2)) + " (lin.reg.)}}")

    # Remove scientific notation along x-axis
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xticks(ns)
    ns_names = []
    for n in ns:
        ns_names.append(f'${n}$')
    ax.set_xticklabels(ns_names)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"\textrm{Error}")

    # Create reference line for desired order of convergence
    if desired_order:
        ns = np.array(ns)
        # line = desired_order * ns + res[1] - 0.5
        line = np.exp(-desired_order * np.log(ns) + res[1] - reference_line_offset)
        ax.plot(ns, line, "--", color="gray",
                label=r"$\textrm{Convergence order " + str(desired_order) + " reference}$")
        ax.legend()


def plot3d(field, title="", latex=False, z_label="z", xs=None, ys=None):
    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        # for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)

    ys = ys if ys is not None else xs
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, field, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("$x$" if latex else "x")
    ax.set_ylabel("$y$" if latex else "y")
    ax.set_zlabel(f"${z_label}$" if latex else z_label)
    return ax


if __name__ == '__main__':
    # Example data with 2nd order convergence
    # ns = [10, 20, 40, 80]
    # errors = [0.0008550438272162397, 0.00021361488748972146, 5.1004121790487744e-05, 1.0208028014102588e-05]

    base = os.getcwd()
    full_path = os.path.join(base, "build/src/cutfem/convergence/errors-d2o1.csv")
    data = np.genfromtxt(full_path, delimiter=",")
    print(data)

    mesh_size = data[:, 0]
    print("mesh", mesh_size)

    u_L2 = data[:, 1]
    u_H1 = data[:, 2]
    u_h1 = data[:, 3]
    p_L2 = data[:, 4]
    p_H1 = data[:, 5]
    p_h1 = data[:, 6]
    ns = 2 ** np.array(range(len(u_L2)))

    skip = 2
    pressure_skip = 3

    u_L2 = u_L2[1 + skip:]
    u_H1 = u_H1[1 + skip:]
    u_h1 = u_h1[1 + skip:]
    p_L2 = p_L2[1 + pressure_skip:]
    p_H1 = p_H1[1 + pressure_skip:]
    p_h1 = p_h1[1 + pressure_skip:]

    print("ns", ns)
    print("u_L2", u_L2)
    print("u_H1", u_H1)
    print("u_h1", u_h1)
    print("p_L2", p_L2)
    print("p_H1", p_H1)
    print("p_h1", p_h1)

    # Velocity plot
    print()
    ns_u = ns[1 + skip:]
    # fig, ((ax_u_L2, ax_u_H1), (ax_p_L2, ax_p_H1)) = plt.subplots(2, 2)

    # Set figure size
    plt.rcParams["figure.figsize"] = (7, 16)

    # VELOCITY
    fig, (ax_u_L2, ax_u_H1, ax_u_h1) = plt.subplots(3, 1, sharex="all")
    fig.subplots_adjust(hspace=0.2)
    fig.suptitle(r"\textrm{\textbf{Velocity} Channel with sphere. AnalyticalSolution=0 inside sphere.}", fontsize=14)
    convergence_plot(ns_u, u_L2, yscale="log2", reference_line_offset=-0.5, xlabel="",
                     title=r"\textrm{L2-norm}",
                     desired_order=1, ax=ax_u_L2)
    convergence_plot(ns_u, u_H1, yscale="log2", reference_line_offset=-0.5, xlabel="",
                     title=r"\textrm{H1-norm}",
                     desired_order=1, ax=ax_u_H1)
    convergence_plot(ns_u, u_h1, yscale="log2", reference_line_offset=-0.5, xlabel="$N$",
                     title=r"\textrm{H1-semi-norm}",
                     desired_order=1, ax=ax_u_h1)

    # PRESSURE
    fig, (ax_p_L2, ax_p_H1, ax_p_h1) = plt.subplots(3, 1, sharex="all")
    fig.subplots_adjust(hspace=0.2)
    fig.suptitle(r"\textrm{\textbf{Pressure:} Channel with sphere.}", fontsize=14)
    # Pressure convergence plot
    ns_p = ns[1 + pressure_skip:]
    print("pressure")
    print(len(ns_p), len(p_L2))
    convergence_plot(ns_p, p_L2, yscale="log2", reference_line_offset=1, xlabel="",
                     title=r"\textrm{L2-norm}",
                     desired_order=1, ax=ax_p_L2)
    convergence_plot(ns_p, p_H1, yscale="log2", reference_line_offset=1, xlabel="",
                     title=r"\textrm{H1-norm}",
                     desired_order=1, ax=ax_p_H1)
    convergence_plot(ns_p, p_h1, yscale="log2", reference_line_offset=1, xlabel="$N$",
                     title=r"\textrm{H1-semi-norm}",
                     desired_order=1, ax=ax_p_h1)

    plt.show()
