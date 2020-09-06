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
    ax.set_title(title)

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

    skip = 0

    mesh_size = data[:, 0]
    print("mesh", mesh_size)
    l2 = data[:, 1]
    print("l2", l2)
    ns = np.ceil(0.41 / mesh_size[1 + skip:])
    print(ns)

    convergence_plot(ns, l2[1 + skip:], yscale="log10", reference_line_offset=0.5, xlabel="$N$",
                     title=r"\textrm{Channel with sphere. AnalyticalSolution=0 inside sphere.}",
                     desired_order=1)
    # convergence_plot(ns, errors, yscale="log2", reference_line_offset=0.5)
