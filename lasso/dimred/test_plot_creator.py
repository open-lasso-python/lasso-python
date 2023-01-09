import math
import os
import random

import numpy as np
import plotly.graph_objects as go

from lasso.dyna.d3plot import ArrayType, D3plot


def create_fake_d3plots(
    path: str,
    element_shell_node_indexes: np.ndarray,
    bend_multiplicator: float,
    n_nodes_x: int = 500,
    n_nodes_y: int = 10,
    n_timesteps: int = 5,
):
    """
    Creates a number of artificial D3plots to be used in testing
    """

    # if bend_multiplicator > 0:
    # bend_loc_x = int(n_nodes_x/10)
    # bend_start = bend_loc_x - int(bend_loc_x/2)
    # bend_end = bend_loc_x + int(bend_loc_x/2)
    # else:
    # bend_loc_x = n_nodes_x - int(n_nodes_x/10)
    # bend_start = bend_loc_x - int(n_nodes_x/20)
    # bend_end = bend_loc_x + int(n_nodes_x/20)

    x_coords = np.arange(n_nodes_x)
    y_coords = np.arange(n_nodes_y)
    # z_bend_mat = np.stack(
    #   [np.array([1+math.sin(x*2*math.pi/(bend_end - bend_start))
    #   for x in range(bend_end - bend_start)]
    # )
    # for _ in range(n_nodes_y)]).reshape(((bend_end - bend_start)*n_nodes_y))

    z_bend_mat = np.stack(
        [
            np.array([1 + math.sin(x * math.pi / n_nodes_x) for x in range(n_nodes_x)])
            for _ in range(n_nodes_y)
        ]
    ).reshape((n_nodes_x * n_nodes_y,))
    node_coordinates = np.zeros((n_nodes_x * n_nodes_y, 3))

    # fill in y coords
    for n in range(n_nodes_y):
        node_coordinates[n * n_nodes_x : n_nodes_x + n * n_nodes_x, 1] = y_coords[n]
        node_coordinates[n * n_nodes_x : n_nodes_x + n * n_nodes_x, 0] = x_coords
    # fill in x coords
    # for n in range(n_nodes_x):
    # node_coordinates[n*n_nodes_y:n_nodes_y+n*n_nodes_y, 0] = x_coords[n]

    node_displacement = np.zeros((n_timesteps, n_nodes_x * n_nodes_y, 3))

    for t in range(n_timesteps):
        node_displacement[t] = node_coordinates
        # node_displacement[t, bend_start*n_nodes_y:bend_end*n_nodes_y, 2] = \
        # z_bend_mat * bend_multiplicator * t
        node_displacement[t, :, 2] = z_bend_mat * bend_multiplicator * t

    # print(node_displacement.shape)

    plot = D3plot()
    plot.arrays[ArrayType.node_displacement] = node_displacement
    plot.arrays[ArrayType.node_coordinates] = node_coordinates
    plot.arrays[ArrayType.element_shell_node_indexes] = element_shell_node_indexes
    plot.arrays[ArrayType.element_shell_part_indexes] = np.full(
        (element_shell_node_indexes.shape[0]), 0
    )

    # we could create an artificial array element_shell_is_alive to test the
    # correct part extraction process not neccessary currently

    os.makedirs(path, exist_ok=True)
    plot.write_d3plot(os.path.join(path, "plot"))
    # plotUtilFunc(node_displacement)


def plot_util_func(xyz_array: np.array):
    trace = go.Scatter3d(
        x=xyz_array[-1, :, 0],
        y=xyz_array[-1, :, 1],
        z=xyz_array[-1, :, 2],
        mode="markers",
        text=np.arange(xyz_array.shape[1]),
    )
    fig = go.Figure([trace])
    fig.show()


def create_element_shell_node_indexes(n_nodes_x: int = 500, n_nodes_y: int = 10) -> np.ndarray:
    """
    returns a element_shell_node_indexes array
    """

    new_shell_node_indexes = np.full(
        ((n_nodes_x - 1) * (n_nodes_y - 1), 4), np.array([0, 1, n_nodes_x + 1, n_nodes_x])
    )
    mod = np.full((4, n_nodes_x - 1), np.arange(n_nodes_x - 1))
    for i in range(n_nodes_y - 1):
        new_shell_node_indexes[(n_nodes_x - 1) * i : (n_nodes_x - 1) + ((n_nodes_x - 1) * i)] += (
            mod + i * n_nodes_x
        ).T

    return new_shell_node_indexes


def create_2_fake_plots(folder: str, n_nodes_x: int, n_nodes_y: int, n_timesteps=5):
    """
    creates 2 faked plots

    Parameters
    ----------
    folder: str
        folder path
    n_nodes_x: int
        how many nodes in x
    n_nodes_y: int
        how many nodes in y
    n_timesteps: int, default: 5
        how many timesteps
    """

    randy_random = random.Random("The_Seed")
    plot_name = "SVDTestPlot{i}"

    element_shell_node_indexes = create_element_shell_node_indexes(
        n_nodes_x=n_nodes_x, n_nodes_y=n_nodes_y
    )

    create_fake_d3plots(
        path=os.path.join(folder, plot_name.format(i="00")),
        element_shell_node_indexes=element_shell_node_indexes,
        bend_multiplicator=5 * (1 + randy_random.random()),
        n_nodes_x=n_nodes_x,
        n_nodes_y=n_nodes_y,
        n_timesteps=n_timesteps,
    )

    create_fake_d3plots(
        path=os.path.join(folder, plot_name.format(i="01")),
        element_shell_node_indexes=element_shell_node_indexes,
        bend_multiplicator=5 * (1 + randy_random.random()),
        n_nodes_x=n_nodes_x,
        n_nodes_y=n_nodes_y,
        n_timesteps=n_timesteps,
    )


def create_50_fake_plots(folder: str, n_nodes_x: int, n_nodes_y: int, n_timesteps=5):
    """
    creates 50 faked plots, 25 bending up, 25 bending down

    Parameters
    ----------
    folder: str
        folder path
    n_nodes_x: int
        how many nodes in x
    n_nodes_y: int
        how many nodes in y
    n_timesteps: int, default: 5
        how many timesteps
    """

    # init random
    randy_random = random.Random("The_Seed")

    plot_name = "SVDTestPlot{i}"

    # doesn't change for each plot with same dimensions, so only created once
    element_shell_node_indexes = create_element_shell_node_indexes(
        n_nodes_x=n_nodes_x, n_nodes_y=n_nodes_y
    )

    # 25 plots bending up
    for i in range(25):
        nr = str(i)
        if i < 10:
            nr = "0" + str(i)
        create_fake_d3plots(
            path=os.path.join(folder, plot_name.format(i=nr)),
            element_shell_node_indexes=element_shell_node_indexes,
            bend_multiplicator=5 * (1 + randy_random.random()),
            n_nodes_x=n_nodes_x,
            n_nodes_y=n_nodes_y,
            n_timesteps=n_timesteps,
        )

    # 25 plots bending down
    for i in range(25):
        create_fake_d3plots(
            path=os.path.join(folder, plot_name.format(i=i + 25)),
            element_shell_node_indexes=element_shell_node_indexes,
            bend_multiplicator=-5 * (1 + randy_random.random()),
            n_nodes_x=n_nodes_x,
            n_nodes_y=n_nodes_y,
            n_timesteps=n_timesteps,
        )


# TODO: Remove after fixing D3plot writing two files issue
# if __name__ == "__main__":
#     create_2_fake_plots("../delteThisPlease/", 200, 10)
