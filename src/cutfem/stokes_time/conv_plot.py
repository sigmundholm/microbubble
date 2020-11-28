import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, eoc_plot

if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]

    skip = 0
    for poly_order in [1]:
        full_path = os.path.join(base, f"build/src/cutfem/stokes_time/errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, :]

        conv_plots(data, head, title=r"$\textrm{Time dep. Stokes, (impl. Euler), element order: (" + str(
            poly_order + 1) + ", " + str(poly_order) + ")}$", domain_length=0.41)
        # plt.savefig(f"figure-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Time dep. Stokes  (impl. Euler) EOC, element order: (" + str(poly_order + 1) + ", " + str(
                     poly_order) + ")}",
                 domain_lenght=0.41, lines_at=np.array([0, 1, 2]) + poly_order)
        # plt.savefig(f"eoc-o{poly_order}.pdf")

    plt.show()
