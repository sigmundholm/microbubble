import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, conv_plots2, eoc_plot, condnum_sensitivity_plot


def convergence_plot_report():
    # A report ready convergence plot
    paths = [os.path.join(base, f"build/src/cutfem/poisson/errors-d2o{d}.csv") for d in [1, 2]]
    plot_for = ["\|u\|_{L^2}", "\|u\|_{H^1}"]
    element_orders = [1, 2]
    conv_plots2(paths, plot_for, element_orders, expected_degrees=[1, 0], domain_length=2.2,
                colors=["mediumseagreen", "darkturquoise"], save_figs=True, font_size=12, label_size="large")


def condition_number_sensitivity_plot():
    # A plot for condition number sensitivity
    path_stab = os.path.join(base, f"build/src/cutfem/poisson/condnums-d2o1r5-stabilized.csv")
    path_non_stab = os.path.join(base, f"build/src/cutfem/poisson/condnums-d2o1r5-nonstabilized.csv")
    condnum_sensitivity_plot(path_stab, path_non_stab, colors=["mediumseagreen", "darkturquoise"],
                             save_figs=True, font_size=12, label_size="large")


if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]

    convergence_plot_report()

    condition_number_sensitivity_plot()
    plt.show()

    skip = 2
    for poly_order in [1, 2]:
        full_path = os.path.join(base, f"build/src/cutfem/poisson/errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, :]

        conv_plots(data, head, title=r"$\textrm{Poisson (CutFEM), element order: " + str(poly_order) + "}$",
                   domain_length=2.2)
        # plt.savefig(f"figure-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Poisson (CutFEM) EOC, element order: " + str(poly_order) + "}",
                 domain_lenght=2.2, lines_at=np.array([0, 1]) + poly_order)
        # plt.savefig(f"eoc-o{poly_order}.pdf")

    plt.show()
