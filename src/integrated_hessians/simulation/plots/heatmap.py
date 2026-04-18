from typing import Optional
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str] = [],
    col_labels: list[str] = [],
    title: str = "",
    ax: Optional[Axes] = None,
    cmap: str = "bwr",
    text: bool = False,
    fig_width=6,
    fig_height=4,
    add_colorbar: bool = True,
) -> Axes:
    """Generic heatmap with borders. matrix shape: (n_rows x n_cols)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(fig_width, fig_height))

    n_rows, n_cols = matrix.shape
    for x in range(n_cols + 1):
        ax.axvline(x - 0.5, color="black", linewidth=1.5)
    for y in range(n_rows + 1):
        ax.axhline(y - 0.5, color="black", linewidth=1.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)

    if text:
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=5,
                )

    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Base")

    # Make 0 white
    vmin = matrix.min()
    vmax = matrix.max()
    # Ensure 0 is within the range to avoid errors
    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        # Fallback if data is all positive or all negative
        norm = None

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", norm=norm)
    # if add_colorbar:
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    return ax
