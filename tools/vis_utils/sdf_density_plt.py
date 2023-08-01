import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def density_func(sdf, beta=0.01):
    alpha = 1 / beta
    return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

if __name__ == '__main__':
    colors = ['#74add1',
        '#abd9e9',
        '#e0f3f8',
        '#ffffbf',
        '#fee090',
        '#fdae61',
        '#f46d43',
        '#d73027',
        '#a50026']
    cmap1 = LinearSegmentedColormap.from_list('mycmap', colors)

    sdf_0 = torch.linspace(3, -2, 10000)
    sdf_1 = torch.linspace(-2, 3, 10000)
    sdf = torch.cat((sdf_0, sdf_1[1:]), dim=0)
    density = density_func(sdf)
    density = density.numpy()
    sdf = sdf.numpy()
    idx = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
    # idx = []
    # for i in range(11):

    x_sdf = [3., 2., 1., 0., -1., -2., -1., 0., 1., 2., 3.]
    # plt.subplot(2, 1, 1)
    plt.ylim(-30, 120)
    plt.xlim(0, 4000)
    plt.yticks([0, 100])
    plt.xticks(idx, x_sdf)
    plt.xlabel('SDF')
    plt.ylabel('Density')
    plt.title('SDF-Density Transformation')
    plt.scatter(list(range(len(density))), density, c=density, cmap=cmap1, s=20)
    ax = plt.gca()
    ax.set_aspect(5)

    # plt.subplot(2, 1, 2)
    # draw_circle = plt.Circle((500, 0), 200, color='orange')
    # draw_arrow = plt.arrow(0, 0, 1000, 0, length_includes_head=True, head_width=20, lw=2)
    # ax2 = plt.gca()
    # ax2.add_artist(draw_circle)
    # ax2.add_artist(draw_arrow)
    # ax2.spines['right'].set_color('none')
    # ax2.spines['top'].set_color('none')
    # ax2.spines['left'].set_color('none')
    # ax2.spines['bottom'].set_color('none')
    # plt.ylim(-200, 200)
    # plt.xlim(0, 1000)
    # plt.yticks([])
    # plt.xticks([])
    plt.show()
    print(1)