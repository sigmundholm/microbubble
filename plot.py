import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import numpy as np

import os


def convergence_plot(ns, errors, yscale="log2", desired_order=2, reference_line_offset=0.5,
                     xlabel="$N$", title=""):
    # Remove small tick lines on the axes, that doesnt have any number with them.
    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0
    matplotlib.rcParams['ytick.minor.size'] = 0
    matplotlib.rcParams['ytick.minor.width'] = 0

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    res = np.polyfit(np.log(ns), np.log(errors), deg=1)
    print("Polyfit:", res)
    print("Order of convergence", -res[0])

    fig, ax = plt.subplots()

    ax.plot(ns, errors, "-o")
    if yscale == "log2":
        ax.set_yscale("log", basey=2)
    else:
        ax.set_yscale("log")

    ax.set_xscale("log")
    ax.set_title(r"\textrm{" + title + r"}\newline \small{\textrm{Convergence order: " + str(-round(res[0],2)) + " (lin.reg.)}}")
    # title(r"""\Huge{Big title !} \newline \tiny{Small subtitle !}""")

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

    plt.show()


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
    ns = [10, 20, 40, 80]
    errors = [0.0008550438272162397, 0.00021361488748972146, 5.1004121790487744e-05, 1.0208028014102588e-05]

    base = os.getcwd()
    full_path = os.path.join(base, "build/src/cutfem/convergence/errors-d2o1.csv")
    data = np.genfromtxt(full_path, delimiter=",")

    skip = 2

    mesh_size = data[:, 0]
    print("mesh", mesh_size)

    l2_u = data[:, 1]
    l2_p = data[:, 2]
    ns = 2 ** np.array(range(len(l2_u)))

    ns = ns[1 + skip:]
    l2_u = l2_u[1 + skip:]
    l2_p = l2_p[1 + skip:]
    print("ns", ns)
    print("l2_u", l2_u)
    print("l2_p", l2_p)

    # Velocity plot
    ns_u = ns
    print()
    convergence_plot(ns_u, l2_u, yscale="log2", reference_line_offset=-0.5, xlabel="$N$",
                     title=r"\textrm{Velocity: Channel with sphere. AnalyticalSolution=0 inside sphere.}",
                     desired_order=1)

    # Pressure convergence plot
    pressure_skip = 1
    ns_p = ns
    print()
    convergence_plot(ns_p[pressure_skip:], l2_p[pressure_skip:], yscale="log2", reference_line_offset=1, xlabel="$N$",
                     title=r"\textrm{Pressure: Channel with sphere. AnalyticalSolution=0 inside sphere.}",
                     desired_order=1)

    # convergence_plot(ns, errors, yscale="log2", reference_line_offset=0.5)
