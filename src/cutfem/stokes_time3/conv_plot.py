import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, eoc_plot, time_error_plots


def time_error_plot():
    for d in element_orders:
        paths = [os.path.join(base, f"build/src/cutfem/stokes_time3", folder, f"errors-time-d2o{d}r{r}.csv") for r in
                 range(3, 7)]
        time_error_plots(paths, data_indices=[3, 4, 5, 6, 7, 8], font_size=12, label_size="large",
                         title=f"Stokes time3 time error, element order ({d + 1}, {d})", save_fig=True,
                         identifier=d)


folder = ""

radius = 0.05
end_time = radius
domain_length = radius
element_orders = [1, 2]

if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]

    time_error_plot()

    skip = 1
    for poly_order in element_orders:
        full_path = os.path.join(base, f"build/src/cutfem/stokes_time3/errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))[1:]
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        # data[:, 1] = data[:, 0]  # TODO only for stationary problems
        data = data[skip:, 1:]

        conv_plots(data, head, title=r"$\textrm{Stokes time3 (cutFEM), element order: (" + str(
            poly_order + 1) + ", " + str(poly_order) + ")}$", domain_length=domain_length)
        plt.savefig(f"figure-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Stokes time3 (cutFEM) EOC, element order: (" + str(poly_order + 1) + ", " + str(
                     poly_order) + ")}",
                 domain_lenght=domain_length, lines_at=np.array([0, 1]) + poly_order)
        plt.savefig(f"eoc-o{poly_order}.pdf")

    plt.show()
