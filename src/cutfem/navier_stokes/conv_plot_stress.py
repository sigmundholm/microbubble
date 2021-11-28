from os.path import join, split

import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, eoc_plot

base = split(split(split(split(__file__)[0])[0])[0])[0]
bm_relative_path = "build/src/cutfem/navier_stokes"
print(base)


def convergence():
    for p in element_order:
        path = join(base, bm_relative_path, f"e-stress-d2o{p}.csv")
        data = np.genfromtxt(path, delimiter=";", skip_header=True)

        grid = data[:, 0]
        print(data)
        for i, name in enumerate([r"\textrm{Drag}", r"\textrm{Lift}"]):
            exact = data[:, 1 + i]
            regular = data[:, 3 + i]
            symmetric = data[:, 5 + i]
            nitsche = data[:, 7 + i]

            new_data = np.zeros((len(exact), 4))
            new_data[:, 0] = grid
            new_data[:, 1] = regular
            new_data[:, 2] = symmetric
            new_data[:, 3] = nitsche
            new_data = np.abs(new_data)

            print(new_data)
            head = ["h", r"\textrm{Regular}", r"\textrm{Symmetric}", r"\textrm{Nitsche}"]
            conv_plots(new_data, head, name, True, domain_length=domain_length, xlabel="N",
                       max_contrast=False, yscale="log")

            # Create a EOC-plot
            eoc_data = new_data[1:, :]
            eoc_plot(eoc_data, head,
                     title=r"\textrm{" + name + " EOC, element order: (" + str(p + 1) + ", " + str(
                         p) + ")}",
                     domain_lenght=domain_length, lines_at=np.array([0, 1, 2]) + p, xlabel="N",
                     max_contrast=False)


element_order = [1]
domain_length = 0.05

if __name__ == '__main__':
    convergence()
    plt.show()
