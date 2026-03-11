import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _(np, plt):
    import matplotlib.colors as mcolors

    X = np.zeros([10, 10])
    Y = np.zeros([10, 10])
    C = np.random.rand(10, 10)
    angle = -45

    for i in range(10):
        for j in range(10):
            X[i, j] = j
            Y[i, j] = i

    # define colormap and normalizer once
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=C.min(), vmax=C.max())

    angle = np.radians(angle)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    fig, ax = plt.subplots()
    gap = 0.05
    for i in range(10):
        for j in range(10):
            if j < i:
                continue
            x_corners = np.array([j+gap, j+1-gap, j+1-gap, j+gap, j+gap])
            y_corners = np.array([i+gap, i+gap, i+1-gap, i+1-gap, i+gap])
            #rotate
            xr = cos_a * x_corners - sin_a * y_corners
            yr = sin_a * x_corners + cos_a * y_corners

            ax.fill(xr, yr, color=cmap(norm(C[i, j])))

    ax.set_aspect('equal')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
