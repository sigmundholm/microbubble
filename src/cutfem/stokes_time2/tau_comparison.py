import os
from os.path import split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.plot import eoc_plot

# Use colours from the Plasma colour map.
cmap = matplotlib.cm.get_cmap("plasma")

base = split(split(split(os.getcwd())[0])[0])[0]


def eoc_plot_after_cut_off_time(build_base, factors, folder_names, end_time, cutoff_time, n_refines):
    for folder, factor in zip(folder_names, factors):
        for poly_order in [1, 2]:
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

                cut_data = data[end_step:, ]

                cut_off_norms = np.sqrt((cut_data ** 2 * tau).sum(axis=0))

                print(cut_data)
                l2 = cut_data[:, 3].max()
                h1 = cut_data[:, 4].max()
                print("cut at ", end_step)
                print("r =", r, ", l2 =", l2, ", h1 =", h1)
                aggregated_data.append(np.array([h, *cut_off_norms[3:], l2, h1]))

            # Use the aggregated_data to calculate a new values for EOC
            create_eoc_plot(end_time,
                            [*head[2:], r'\|u\|_{l^\infty L^2}', r'\|u\|_{l^\infty H^1}'], aggregated_data,
                            poly_order, factor, cutoff_time)


def create_eoc_plot(end_time, head, aggregated_data, poly_order, tau_factor, cutoff_time):
    data_cols = np.array(aggregated_data)

    print("\n", head)
    print("order:", poly_order, ", factor =", tau_factor)
    print(data_cols)
    # Let time be on the first axis

    xlabel = "M"

    eoc_plot(data_cols, head,
             title=r"\textrm{Stokes Equations, EOC, element order $(" + str(poly_order + 1) + "," +
                   str(poly_order) + r") $, $\tau=h/" + str(
                 tau_factor) + r"$, for $t\geq " + str(cutoff_time / end_time) + r"\, T$}",
             domain_lenght=end_time, lines_at=np.array([1, 2, 3]), xlabel=xlabel)
    plt.savefig(f"eoc-cut-o{poly_order}-h_{tau_factor}tau.pdf")


if __name__ == '__main__':
    build_base = os.path.join(base, "build/src/cutfem/stokes_time2")
    factors = [1, 2]
    folder_names = ["e_bdf2_bdf1_tau_h_01", "e_bdf2_bdf1_tau_h_02"]

    radius = 0.0625
    end_time = radius
    cutoff_time = end_time / 4
    n_refines = range(3, 8)

    eoc_plot_after_cut_off_time(build_base, factors, folder_names, end_time, cutoff_time, n_refines)
    plt.show()
