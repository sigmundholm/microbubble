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
    ax.set_title(r"\textrm{" + title + r"}\newline \small{\textrm{Convergence order: " + str(
        -round(res[0], 2)) + " (lin.reg.)}}")
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


def add_convergence_line(ax, ns, errors, yscale="log2", xlabel="$N$", name="", color=None, regression=True):
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

    label = f'${name}$' if not regression else f'${name}: {abs(round(res[0], 3))}$'
    ax.plot(ns, errors, "-o", label=label, color=color)
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


def if_latex(latex: bool):
    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        # for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)


def plot3d(field, title="", latex=False, z_label="z", xs=None, ys=None):
    if_latex(latex)

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


def conv_plots(data, columns, title="", latex=True, domain_length=1, xlabel="N"):
    if_latex(latex)

    mesh_size = data[:, 0]
    ns = list(map(round, domain_length / mesh_size))

    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0
    matplotlib.rcParams['ytick.minor.size'] = 0
    matplotlib.rcParams['ytick.minor.width'] = 0

    cmap = matplotlib.cm.get_cmap("plasma")

    fig, ax = plt.subplots()
    for k, (col_name, data_col) in enumerate(zip(columns[1:], [data[:, i] for i in range(1, data.shape[1])])):
        print()
        print(col_name, data_col)
        ax = add_convergence_line(ax, ns, data_col, "log2", name=col_name, xlabel=f"${xlabel}$",
                                  color=cmap(k / (len(columns) - 1 - int(len(columns) > 7))))
    ax.set_title(title)


def eoc_plot(data, columns, title="", domain_lenght=1, latex=True, lines_at=None, xlabel="N"):
    if_latex(latex)

    mesh_size = data[:, 0]
    ns = domain_lenght / mesh_size

    # Remove small tick lines on the axes, that doesnt have any number with them.
    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0
    matplotlib.rcParams['ytick.minor.size'] = 0
    matplotlib.rcParams['ytick.minor.width'] = 0

    cmap = matplotlib.cm.get_cmap("plasma")

    fig, ax = plt.subplots()
    ax.set_xscale("log")

    if lines_at is not None:
        for line_val in lines_at:
            ax.plot([ns[1], ns[-1]], [line_val, line_val], linestyle='--', linewidth=1, color="gray")

    for k, (col_name, data_col) in enumerate(zip(columns[1:], [data[:, i] for i in range(1, data.shape[1])])):
        eoc = np.log(data_col[:-1] / data_col[1:]) / np.log(mesh_size[:-1] / mesh_size[1:])
        print(col_name, eoc)

        ax.plot(ns[1:], eoc, label=r"${" + col_name + "}$", linestyle='--', marker='.',
                color=cmap(k / (len(columns) - 1 - int(len(columns) > 7))))

    # Remove scientific notation along x-axis
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xticks(ns[1:])
    ns_names = [f'${int(round(n, 0))}$' for n in ns[1:]]
    ax.set_xticklabels(ns_names)

    ax.set_title(title)
    ax.set_xlabel(f"${xlabel}$")
    ax.set_ylabel(r"\textrm{EOC}")
    ax.legend()

    return ax


def conv_plots2(paths, norm_names, element_orders, expected_degrees, domain_length=1.0,
                colors=None, save_figs=False, font_size=10, label_size='medium', skip=0,
                ylabel=None, guess_degree=True):
    """
    Creates convergence plot for the report. One plot for each norm, then one convergence
    line for each element order.

    :param paths:
    :param norm_names:
    :param element_orders:
    :param expected_degrees:
    :param domain_length:
    :param colors:
    :return:
    """
    if_latex(True)

    dfs = []
    head = ""
    mesh_size = []
    for full_path in paths:
        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)

        mesh_size = data[:, 0]
        dfs.append(data)

    ns = list(map(int, domain_length / mesh_size))[skip:]

    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0
    matplotlib.rcParams['ytick.minor.size'] = 0
    matplotlib.rcParams['ytick.minor.width'] = 0

    cmap = matplotlib.cm.get_cmap("plasma")
    colors = colors if colors is not None else [cmap((i + 1) / (len(element_orders) + 1)) for i in
                                                range(len(element_orders))]

    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)
    plt.rcParams.update({'axes.labelsize': label_size})

    for i, norm_name in enumerate(norm_names):
        fig, ax = plt.subplots()
        for deg_index, degree in enumerate(element_orders):
            data_column = head.index(norm_name)
            errors = dfs[deg_index][skip:, data_column]
            color = None if colors is None else colors[deg_index]
            ax = add_convergence_line(ax, ns, errors, yscale="log", name=f"k={degree}", color=color, regression=False)
            if guess_degree:
                guess = expected_degrees[i] + degree
            else:
                guess = expected_degrees[i][deg_index]

            add_conv_triangle(ax, guess, color, errors[-1], ns[-2:])

        ylabel_text = f"${norm_name}".replace("u", "u - u_h")[:-1] + r"(\Omega)}$" if ylabel is None else ylabel
        ax.set_ylabel(ylabel_text)

        if save_figs:
            plt.savefig(f"conv-norm-{i}.svg")
            plt.savefig(f"conv-norm-{i}.pdf")


def add_conv_triangle(ax, degree, color, right_error, ns):
    left_n, right_n = ns[0] * 1.1, ns[1] / 1.1
    bottom_value = right_error / 1.05
    top_value = bottom_value * 2 ** (degree * np.log2(right_n / left_n))  # from EOC formula

    linestyle = "dashed"
    linewidth = 1
    # Horisontal line
    ax.plot([left_n, right_n], [bottom_value, bottom_value], color=color, linestyle=linestyle, linewidth=linewidth)
    ax.text(ns[0] * 1.4, bottom_value / 1.5, f"$1$", color=color)

    # Vertical line
    ax.plot([left_n, left_n], [bottom_value, top_value], color=color, linestyle=linestyle, linewidth=linewidth)
    degree_text_x = ns[0] / 1.05 if degree < 0 else ns[0]
    ax.text(degree_text_x, bottom_value * 2 ** (degree / 4), f"${degree}$", color=color)

    # Diagonal line
    ax.plot([left_n, right_n], [top_value, bottom_value], color=color, linestyle=linestyle, linewidth=linewidth)


def condnum_sensitivity_plot(path_stabilized, path_nonstabilized, colors=None, save_figs=True,
                             font_size=10, label_size='medium', errors=False):
    """

    :param path_stabilized:
    :param path_nonstabilized:
    :param colors:
    :param save_figs:
    :param font_size:
    :param label_size:
    :param errors: plotting a sensitivity plot of the condition number when set to
    False, and of the errors when True.
    :return:
    """
    if_latex(True)
    data_stab = np.genfromtxt(path_stabilized, delimiter=",", skip_header=True)
    data_non_stab = np.genfromtxt(path_nonstabilized, delimiter=",", skip_header=True)

    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)
    plt.rcParams.update({'axes.labelsize': label_size})

    fig, ax = plt.subplots()
    data_columns = [2, 3] if errors else [1]
    first_axis = data_stab[:, 0] / data_stab[:, 0].max()

    # Plot the L^1 and the H^1 errors.
    if errors:
        # H^1
        ax.plot(first_axis, data_stab[:, 3], color=colors[3], label=r"\textrm{$H^1$ (stabilized)}")
        ax.plot(first_axis, data_non_stab[:, 3], color=colors[2], label=r"\textrm{$H^1$ (not stabilized)}")

        # L^2
        ax.plot(first_axis, data_stab[:, 2], color=colors[1], label=r"\textrm{$L^2$ (stabilized)}")
        ax.plot(first_axis, data_non_stab[:, 2], color=colors[0], label=r"\textrm{$L^2$ (not stabilized)}")

        ax.set_ylabel(r"$\|u - u_h\|_{L^2(\Omega_F)} \qquad \|u - u_h\|_{H^1(\Omega_F)}$")

    # Plot the condition numbers
    else:
        ax.plot(first_axis, data_stab[:, 1], color=colors[2], label=r"\textrm{Stabilized}")

        ax.plot(first_axis, data_non_stab[:, 1], color=colors[3], label=r"\textrm{Not stabilized}")
        ax.set_ylabel("$\\kappa(A)$")

    ax.set_yscale("log", base=10)

    ax.set_xlabel("$\delta$")
    ax.legend()

    if save_figs:
        plt.savefig(f"sensitivity-{'error' if errors else 'condnum'}.svg")


def time_error_plots(paths, data_indices, title="", save_fig=True, identifier=1, font_size=10,
                     label_size='medium'):
    if_latex(True)

    dfs = []
    head = ""
    for full_path in paths:
        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        dfs.append(data)

    cmap = matplotlib.cm.get_cmap("plasma")

    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0
    matplotlib.rcParams['ytick.minor.size'] = 0
    matplotlib.rcParams['ytick.minor.width'] = 0

    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)
    plt.rcParams.update({'axes.labelsize': label_size})

    for data_index in data_indices:
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        for i, df in enumerate(dfs):
            error = df[:, data_index]

            time_steps = df[:, 0]
            taus = df[:, 1]
            times = time_steps * taus

            ax.plot(times, error, label=f"$M={len(time_steps) - 1}$", linestyle="--", marker=".",
                    color=cmap(i / len(paths)))

        ax.set_xlabel(r"$t\, [s]$")
        ax.set_ylabel(f"${head[data_index].replace('u', 'u-u_h')}$")
        ax.set_title(r"$\textrm{" + f"{title}" + "}$")
        ax.legend(loc='upper right')

        if save_fig:
            plt.savefig(f"{'-'.join(map(str.lower, title.split()[:2]))}-{identifier}-{data_index}.pdf")


def eoc_plot_after_cut_off_time(build_base, factors, folder_names, element_orders, end_time, cutoff_time, n_refines,
                                columns_idx, max_norm_idx=(), max_norm_names=()):
    for folder, factor in zip(folder_names, factors):
        for poly_order in element_orders:
            print("\nfactor =", factor, ", order =", poly_order)
            # Errors calculated from the cutoff time
            aggregated_data = []
            head = []

            for r in n_refines:
                full_path = os.path.join(build_base, folder, f"errors-time-d2o{poly_order}r{r}.csv")
                print(full_path)

                head = list(map(str.strip, open(full_path).readline().split(",")))

                data = np.genfromtxt(full_path, delimiter=",", skip_header=True)

                tau = end_time / (pow(2, r - 1) * factor)
                h = factor * tau
                end_step = int(cutoff_time / tau)

                cut_data = data[end_step:, :]
                print(cut_data)

                cut_off_norms = np.sqrt((cut_data ** 2 * tau).sum(axis=0))

                max_norms = [cut_data[:, i].max() for i in max_norm_idx]
                aggregated_data.append(np.array([h, *cut_off_norms[columns_idx], *max_norms]))

            # Use the aggregated_data to calculate a new values for EOC
            data_cols = np.array(aggregated_data)
            print("\n", head)
            print("order:", poly_order, ", factor =", factor)
            xlabel = "M"

            cut_head = ["h", *[head[i] for i in columns_idx], *max_norm_names]
            print(cut_head)
            eoc_plot(data_cols, cut_head,
                     title=r"\textrm{Heat Equation (CutFEM) EOC, $k=" + str(poly_order) + r"$, $\tau=h/" + str(
                         factor) + r"$, for $t\geq " + str(cutoff_time / end_time) + r"\, T$}",
                     domain_lenght=end_time, lines_at=np.array([1, 2, 3]), xlabel=xlabel)
            plt.savefig(f"eoc-cut-o{poly_order}-h_{factor}tau.pdf")


if __name__ == '__main__':
    base = os.getcwd()

    poly_order = 2
    full_path = os.path.join(base, f"build/src/streamline_diffusion/errors-o{poly_order}-eps=0.100000.csv")

    head = list(map(str.strip, open(full_path).readline().split(",")))
    data = np.genfromtxt(full_path, delimiter=",", skip_header=True)

    conv_plots(data, head, title=f"Polynomial order: {poly_order}")
