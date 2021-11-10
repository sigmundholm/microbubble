import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, eoc_plot

if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]
    domain_length = 0.2

    skip = 0
    for poly_order in [2]:
        full_path = os.path.join(base, f"build/src/cutfem/stokes_time3/errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))[1:-2]
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        print(data)
        data[:, 1] = data[:, 0] # TODO only for stationary problems

        print(data)
        data = data[skip:, 1:-2]

        conv_plots(data, head, title=r"$\textrm{Generalized Stokes (cutFEM), element order: (" + str(
            poly_order + 1) + ", " + str(poly_order) + ")}$", domain_length=domain_length)
        plt.savefig(f"figure-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Generalized Stokes (cutFEM) EOC, element order: (" + str(poly_order + 1) + ", " + str(
                     poly_order) + ")}",
                 domain_lenght=domain_length, lines_at=np.array([0, 1, 2]) + poly_order)
        plt.savefig(f"eoc-o{poly_order}.pdf")

    plt.show()
