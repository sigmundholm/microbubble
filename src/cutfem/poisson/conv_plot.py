import os
from os.path import split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.plot import conv_plots, conv_plots2, eoc_plot, condnum_sensitivity_plot

# Use colours from the Plasma colour map.
cmap = matplotlib.cm.get_cmap("plasma")
color0 = cmap(0.3)
color1 = cmap(0.5)
color2 = cmap(0.7)
color3 = cmap(0.9)


def convergence_plot_report():
    # A report ready convergence plot
    paths = [os.path.join(base, f"build/src/cutfem/poisson/errors-d2o{d}.csv") for d in [1, 2]]
    plot_for = ["\|u\|_{L^2}", "\|u\|_{H^1}"]
    element_orders = [1, 2]
    conv_plots2(paths, plot_for, element_orders, expected_degrees=[1, 0], domain_length=domain_length,
                colors=[color2, color1], save_figs=True, font_size=12, label_size="large", skip=2)


def condition_number_sensitivity_plot():
    # A plot for condition number sensitivity
    path_stab = os.path.join(base, f"build/src/cutfem/poisson/condnums-d2o1r5-stabilized.csv")
    path_non_stab = os.path.join(base, f"build/src/cutfem/poisson/condnums-d2o1r5-nonstabilized.csv")
    condnum_sensitivity_plot(path_stab, path_non_stab, colors=[color0, color1, color2, color3],
                             save_figs=True, font_size=12, label_size="large", errors=False)
    condnum_sensitivity_plot(path_stab, path_non_stab, colors=[color0, color1, color2, color3],
                             save_figs=True, font_size=12, label_size="large", errors=True)


def condition_number_plot():
    paths = [os.path.join(base, f"build/src/cutfem/poisson/errors-d2o{d}.csv") for d in [1, 2]]
    plot_for = ["\kappa(A)"]
    element_orders = [1, 2]
    conv_plots2(paths, plot_for, element_orders, expected_degrees=[-2, -2], domain_length=domain_length,
                colors=[color2, color1], save_figs=True, font_size=12, label_size="large",
                skip=4, ylabel=f"${plot_for[0]}$", guess_degree=False)


# Plot settings
domain_length = 1

if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]

    convergence_plot_report()

    condition_number_sensitivity_plot()

    condition_number_plot()

    skip = 2
    for poly_order in [1, 2]:
        full_path = os.path.join(base, f"build/src/cutfem/poisson/errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, :]

        conv_plots(data, head, title=r"$\textrm{Poisson (CutFEM), element order: " + str(poly_order) + "}$",
                   domain_length=domain_length)
        # plt.savefig(f"figure-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Poisson (CutFEM) EOC, element order: " + str(poly_order) + "}",
                 domain_lenght=domain_length, lines_at=np.array([0, 1]) + poly_order)
        # plt.savefig(f"eoc-o{poly_order}.pdf")

    plt.show()
