import os
from os.path import join, split

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.plot import if_latex

if_latex(True)

# Use colours from the Plasma colour map.
cmap = matplotlib.cm.get_cmap("plasma")
color0 = cmap(0.3)
color1 = cmap(0.5)
color2 = cmap(0.7)
color3 = cmap(0.9)


def betatest_plot(base, build_path, data_indices, element_orders, lines_at, font_size=12, tight=False):
    values2files = locate_files(base, build_path)
    head, data_dic = read_data(base, build_path, values2files)

    colours = [cmap((i + 1) / (len(head) + 1)) for i in range(len(head))]

    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)

    for p in element_orders:
        fig, ax = plt.subplots()
        if lines_at is not None:
            for line_val in lines_at:
                ax.plot([data_dic[0][0], data_dic[-1][0]], [line_val, line_val], linestyle='--', linewidth=1,
                        color="gray")

        create_pretty_beta_plot(ax, head, data_dic, data_indices, colours)
        options = {'bbox_inches': 'tight'} if tight else {}
        plt.savefig(f"betatest-o{p}.pdf", **options)


def locate_files(base, build_path):
    beta_sep = "="
    value2file = []
    # TODO sort by element order too
    # for p in element_orders:
    #   if f"errors-d2o{p}" in filename ....
    for file in os.listdir(join(base, build_path)):
        if beta_sep in file:
            value = file.split(beta_sep)[1][:-4]
            value2file.append((float(value), file))
    value2file.sort(key=lambda x: x[0])
    return value2file


def read_data(base, build_path, value2files):
    head = ""
    beta2data = []
    for i, (beta, file_name) in enumerate(value2files):
        full_path = join(base, build_path, file_name)
        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        beta2data.append((beta, data))
    return head, beta2data


def compute_eoc():
    pass


def create_pretty_beta_plot(ax, head, beta2data, data_indices, colours):
    for data_index in data_indices:
        norm_name = head[data_index]
        betas = []
        eocs = []
        for beta, data in beta2data:
            mesh_size = data[:, 0]

            data_col = data[:, data_index]
            eoc = np.log(data_col[:-1] / data_col[1:]) / np.log(mesh_size[:-1] / mesh_size[1:])
            betas.append(beta)
            eocs.append(eoc[-1])

        ax.plot(betas, eocs, label=f"${norm_name}$", linestyle="--", marker=".", color=colours[data_index])

    ax.legend()


relative_path = "build/src/cutfem/heat_eqn"
degrees = [1, 2]
domain_length = 1

if __name__ == '__main__':
    base = split(split(split(os.getcwd())[0])[0])[0]

    head_indices = [2, 3, 5]

    betatest_plot(base, relative_path, head_indices, [1], lines_at=[1, 2])
    plt.show()
