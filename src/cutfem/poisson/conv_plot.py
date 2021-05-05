import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, eoc_plot

if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]

    skip = 0
    for poly_order in [1, 2]:
        full_path = os.path.join(base, f"build/src/cutfem/poisson/errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, :]

        conv_plots(data, head, title=r"$\textrm{Poisson (CutFEM), element order: " + str(poly_order) + "}$", domain_length=2)
        # plt.savefig(f"figure-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Poisson (CutFEM) EOC, element order: " + str(poly_order) + "}",
                 domain_lenght=2, lines_at=np.array([0, 1]) + poly_order)
        # plt.savefig(f"eoc-o{poly_order}.pdf")

    plt.show()
