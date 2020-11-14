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


def add_convergence_line(ax, ns, errors, yscale="log2", xlabel="$N$", name=""):
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
    print("Polyfit:", name, res)
    print("Order of convergence", -res[0])

    ax.plot(ns, errors, "-o", label=f'${name}: {abs(round(res[0], 3))}$')
    if yscale == "log2":
        ax.set_yscale("log", base=2)
    else:
        ax.set_yscale("log")

    ax.set_xscale("log")

    # Remove scientific notation along x-axis
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xticks(ns)
    ns_names = []
    for n in ns:
        ns_names.append(f'${round(n, 3)}$')
    ax.set_xticklabels(ns_names)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"\textrm{Error}")

    ax.legend()

    return ax


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


def conv_plots(data, columns, title="", latex=False):
    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    print(columns)
    first_axis_name = columns[0]
    mesh_size = data[:, 0]
    # ns = 1 / mesh_size  # TODO hardcodet inn 1 for domenestørrelsen her, fix...

    fig, ax = plt.subplots()
    for col_name, data_col in zip(columns[1:], [data[:, i] for i in range(1, data.shape[1])]):
        print()
        print(col_name, data_col)
        ax = add_convergence_line(ax, mesh_size, data_col, "log2", name=col_name, xlabel="$h$")
    ax.set_title(title)


if __name__ == '__main__':
    base = os.getcwd()

    poly_order = 2
    full_path = os.path.join(base, f"build/src/streamline_diffusion/errors-o{poly_order}-eps=0.100000.csv")

    head = list(map(str.strip, open(full_path).readline().split(",")))
    data = np.genfromtxt(full_path, delimiter=",", skip_header=True)

    conv_plots(data, head, title=f"Polynomial order: {poly_order}")
