import os
from os.path import split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.plot import conv_plots, conv_plots2, eoc_plot, condnum_sensitivity_plot, time_error_plots

# Use colours from the Plasma colour map.
cmap = matplotlib.cm.get_cmap("plasma")
color0 = cmap(0.3)
color1 = cmap(0.5)
color2 = cmap(0.7)
color3 = cmap(0.9)

folder = ""


def convergence_plot_report():
    # A report ready convergence plot
    paths = [os.path.join(base, f"build/src/cutfem/heat_eqn", folder, f"errors-d2o{d}.csv") for d in [1, 2]]
    plot_for = ["\|u\|_{L^2}", "\|u\|_{H^1}", "\|u\|_{l^\infty L^2}", "\|u\|_{l^\infty H^1}"]
    element_orders = [1, 2]
    conv_plots2(paths, plot_for, element_orders, expected_degrees=[[2, 3], [1, 2], [2, 3], [1, 2]], domain_length=2.2,
                colors=[color2, color1, color0], save_figs=True, font_size=12, label_size="large", skip=0, guess_degree=False)


def condition_number_sensitivity_plot():
    # A plot for condition number sensitivity
    path_stab = os.path.join(base, f"build/src/cutfem/heat_eqn", folder, "condnums-d2o1r5-stabilized.csv")
    path_non_stab = os.path.join(base, f"build/src/cutfem/heat_eqn", folder, "condnums-d2o1r5-nonstabilized.csv")
    condnum_sensitivity_plot(path_stab, path_non_stab, colors=[color0, color1, color2, color3],
                             save_figs=True, font_size=12, label_size="large", errors=False)
    condnum_sensitivity_plot(path_stab, path_non_stab, colors=[color0, color1, color2, color3],
                             save_figs=True, font_size=12, label_size="large", errors=True)


def condition_number_plot():
    paths = [os.path.join(base, f"build/src/cutfem/heat_eqn", folder, f"errors-d2o{d}.csv") for d in [1, 2]]
    plot_for = ["\kappa(A)"]
    element_orders = [1, 2]
    conv_plots2(paths, plot_for, element_orders, expected_degrees=[-2, -2], domain_length=2.2,
                colors=[color2, color1], save_figs=True, font_size=12, label_size="large",
                skip=4, ylabel=f"${plot_for[0]}$", guess_degree=False)


def time_error_plot():
    for d in [1, 2]:
        paths = [os.path.join(base, f"build/src/cutfem/heat_eqn", folder, f"errors-time-d2o{d}r{r}.csv") for r in range(2, 8)]
        time_error_plots(paths, end_time=1.1, data_indices=[1, 2], font_size=12, label_size="large",
                         title=f"Heat equation time error, element order {d}", save_fig=True, identifier=d)


if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]
    # time_error_plot()

    # convergence_plot_report()

    # plt.show()
    # exit()
    # condition_number_sensitivity_plot()

    # condition_number_plot()
    # plt.show()
    # exit()

    skip = 0
    domain_length = 1.1
    xlabel = "M"
    for poly_order in [1, 2]:
        full_path = os.path.join(base, f"build/src/cutfem/heat_eqn", folder, f"errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))[1:-1]
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, 1:-1]

        conv_plots(data, head, title=r"$\textrm{Heat Equation (CutFEM), element order: " + str(poly_order) + "}$",
                   domain_length=domain_length, xlabel=xlabel)
        plt.savefig(f"figure-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Heat Equation (CutFEM) EOC, element order: " + str(poly_order) + "}",
                 domain_lenght=domain_length, lines_at=np.array([1, 2]), xlabel=xlabel)
        plt.savefig(f"eoc-o{poly_order}.pdf")

    plt.show()
