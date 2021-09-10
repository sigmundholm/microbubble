import os
from os.path import split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.plot import conv_plots, conv_plots2, eoc_plot, condnum_sensitivity_plot, time_error_plots

# Use colours from the Plasma colour map.
cmap = matplotlib.cm.get_cmap("plasma")

base = split(split(split(os.getcwd())[0])[0])[0]


def get_data():
    build_base = os.path.join(base, "build/src/cutfem/heat_eqn")
    factors = [1, 2, 4]
    folder_names = [f"e_bdf2_bdf1_error_h_tau_0{factor}" for factor in factors]

    h = 0.55
    end_time = 1.1

    cutoff_time = h / 2
    n_refines = range(3, 8)

    for folder, factor in zip(folder_names, factors):
        for poly_order in [1, 2]:
            # Errors calculated from the cutoff time
            aggregated_data = []
            head = []

            for r in n_refines:
                full_path = os.path.join(base, f"build/src/cutfem/heat_eqn", folder,
                                         f"errors-time-d2o{poly_order}r{r}.csv")

                head = list(map(str.strip, open(full_path).readline().split(",")))

                data = np.genfromtxt(full_path, delimiter=",", skip_header=True)

                tau = end_time / (pow(2, r - 1) * factor)
                h = factor * tau
                end_step = int(cutoff_time / tau)

                cut_data = data[end_step:, ]

                cut_off_norms = np.sqrt((cut_data ** 2 * tau).sum(axis=0))
                aggregated_data.append(np.array([h, *cut_off_norms[1:-1]]))

                print("\ntau =", tau)
                print("end_step =", end_step)

            # Use the aggregated_data to calculate a new values for EOC
            calc_eocs(head[1:-1], aggregated_data, poly_order, factor)


def calc_eocs(head, aggregated_data, poly_order, tau_factor):
    data_cols = np.array(aggregated_data)
    mesh_size = data_cols[:, 0]

    end_time = 1.1
    ns = end_time / mesh_size
    xlabel = "M"

    eoc_plot(data_cols, head,
             title=r"\textrm{Heat Equation (CutFEM) EOC, $k=" + str(poly_order) + r"$, $\tau=h/" + str(
                 tau_factor) + "$}",
             domain_lenght=end_time, lines_at=np.array([1, 2, 3]), xlabel=xlabel)
    plt.savefig(f"eoc-cut-o{poly_order}-h_{tau_factor}tau.pdf")


get_data()

plt.show()
