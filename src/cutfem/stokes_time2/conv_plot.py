import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, eoc_plot

if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]

    skip = 0
    for poly_order in [1]:
        full_path = os.path.join(base, f"build/src/cutfem/stokes_time2/errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))[1:]
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, 1:]

        conv_plots(data, head, title=r"$\textrm{Time dep. Stokes (BDF-2), element order: (" + str(
            poly_order + 1) + ", " + str(poly_order) + ")}$", domain_length=0.205 * 2)
        plt.savefig(f"bdf2-error-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Time dep. Stokes (BDF-2) EOC, element order: (" + str(poly_order + 1) + ", " + str(
                     poly_order) + ")}",
                 domain_lenght=0.205 * 2, lines_at=np.array([1, 2]))
        plt.savefig(f"bdf2-eoc-o{poly_order}.pdf")

    plt.show()