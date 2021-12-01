from os.path import join, split

import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, eoc_plot

base = split(split(split(split(__file__)[0])[0])[0])[0]
bm_relative_path = "build/src/cutfem/poisson"


def convergence():
    for p in element_order:
        path = join(base, bm_relative_path, f"e-flux-d2o{p}.csv")

        head = list(map(str.strip, open(path).readline().split(";")))
        data = np.genfromtxt(path, delimiter=";", skip_header=True)

        head = ["h", *[r"\textrm{" + name.replace('_', ' ').title() + "}" for name in head[1:]]]
        conv_plots(data, head, r"\textrm{Poisson boundary flux error, element order: " + str(p) + "}",
                   True, domain_length=domain_length, xlabel="N")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Poisson boundary flux EOC, element order: " + str(p) + "}",
                 domain_lenght=domain_length, lines_at=np.array([0, 1, 2]) + p, xlabel="N")


element_order = [1, 2]
domain_length = 1

if __name__ == '__main__':
    convergence()
    plt.show()
