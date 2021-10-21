import os
from os.path import split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.plot import eoc_plot_after_cut_off_time

# Use colours from the Plasma colour map.
cmap = matplotlib.cm.get_cmap("plasma")

base = split(split(split(os.getcwd())[0])[0])[0]


if __name__ == '__main__':
    build_base = os.path.join(base, "build/src/cutfem/stokes_time2")
    factors = [1]
    folder_names = [""]
    element_orders = [2]

    radius = 0.15
    end_time = radius
    cutoff_time = end_time / 4
    n_refines = range(3, 7)

    data_columns = [3, 4, 5, 6, 7, 8]
    max_norm_indices = [3, 4]
    max_norm_names = [r'\|u\|_{l^\infty L^2}', r'\|u\|_{l^\infty H^1}']
    eoc_plot_after_cut_off_time(build_base, factors, folder_names, element_orders, end_time, cutoff_time, n_refines,
                                data_columns, max_norm_indices, max_norm_names)
    plt.show()
