from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
from captum.attr import IntegratedGradients
from path_explain import PathExplainerTorch
import marimo as mo

elev_slider = mo.ui.slider(
    start=-90,
    stop=90,
    value=20,
    label="Elevation",
    step=10,
    show_value=True,
)
azim_slider = mo.ui.slider(
    start=0,
    stop=360,
    value=340,
    label="Azimuth",
    step=10,
    show_value=True,
)
slider_input_x = mo.ui.slider(
    start=0.0,
    stop=1.0,
    value=1.0,
    label="input x:",
    step=0.1,
    show_value=True,
)
slider_input_y = mo.ui.slider(
    start=0.0,
    stop=1.0,
    value=1.0,
    label="input y:",
    step=0.1,
    show_value=True,
)
slider_baseline_x = mo.ui.slider(
    start=0.0,
    stop=1.0,
    value=0.5,
    label="baseline x:",
    step=0.1,
    show_value=True,
)
slider_baseline_y = mo.ui.slider(
    start=0.0,
    stop=1.0,
    value=0.5,
    label="baseline y:",
    step=0.1,
    show_value=True,
)


sample_x_range = np.linspace(0, 1, 50)
sample_y_range = np.linspace(0, 1, 50)


def surface_from_function(
    f,
    elev,
    azim,
) -> tuple[Figure, Axes]:

    # Generate x and y values

    # Create a meshgrid for 3D plotting
    default_surface_x, default_surface_y = np.meshgrid(sample_x_range, sample_y_range)
    default_surface_z = f(default_surface_x, default_surface_y)

    # Use the Object-Oriented (OO) API
    ax: Axes3D
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))  # type: ignore

    # Plot the surface
    surf = ax.plot_surface(
        default_surface_x,
        default_surface_y,
        default_surface_z,
        cmap="viridis",
        edgecolor="none",
    )

    # Add labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # set camera
    ax.view_init(elev=elev, azim=azim)

    return fig, ax
