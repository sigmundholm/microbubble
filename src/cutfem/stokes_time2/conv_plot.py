import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots, eoc_plot, time_error_plots, conv_plots2


def convergence_plot_report():
    # A report ready convergence plot
    paths = [os.path.join(base, f"build/src/cutfem/stokes_time2", folder, f"errors-d2o{d}.csv") for d in [1, 2]]
    plot_for = [r"\|u\|_{L^2L^2}", r"\|u\|_{L^2H^1}", "\|u\|_{l^\infty L^2}", "\|u\|_{l^\infty H^1}"]
    element_orders = [1, 2]
    conv_plots2(paths, plot_for, element_orders, expected_degrees=[[2, 2], [1, 2], [2, 2], [1, 2]],
                domain_length=domain_length, save_figs=True, font_size=12, label_size="large", skip=0,
                guess_degree=False)


def time_error_plot():
    for d in [1, 2]:
        paths = [os.path.join(base, f"build/src/cutfem/stokes_time2", folder, f"errors-time-d2o{d}r{r}.csv") for r in
                 range(3, 8)]
        time_error_plots(paths, data_indices=[3, 4, 5, 6, 7, 8], font_size=12, label_size="large",
                         title=f"Stokes Equations time error, element order ({d + 1}, {d})", save_fig=True,
                         identifier=d)


# Plot settings
folder = ""
radius = 0.05
end_time = radius
domain_length = radius
xlabel = "M"

if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]

    time_error_plot()
    # convergence_plot_report()

    # time_error_plot()

    skip = 0
    for poly_order in [1, 2]:
        full_path = os.path.join(base, "build/src/cutfem/stokes_time2", folder, f"errors-d2o{poly_order}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))[1:]
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, 1:]

        conv_plots(data, head, title=r"$\textrm{Time dep. Stokes, element order: (" + str(
            poly_order + 1) + ", " + str(poly_order) + ")}$", domain_length=domain_length, xlabel=xlabel)
        plt.savefig(f"bdf2-error-o{poly_order}.pdf")

        # Create a EOC-plot
        eoc_plot(data, head,
                 title=r"\textrm{Time dep. Stokes EOC, element order: (" + str(poly_order + 1) + ", " + str(
                     poly_order) + ")}",
                 domain_lenght=domain_length, lines_at=np.array([1, 2]), xlabel=xlabel)
        plt.savefig(f"bdf2-eoc-o{poly_order}.pdf")

    plt.show()
