from typing import Optional

from beartype import beartype
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import torch
from integrated_hessians.simulation import NUCLEOTIDE_ORDER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import jaxtyping as jx


def plot_heatmap(
    matrix: np.ndarray,
    col_labels: list[str],
    row_labels: list[str],
    title: str,
    ax: Optional[Axes] = None,
    cmap: str = "bwr",
    text: bool = False,
    add_colorbar: bool = True,
) -> Axes:
    """Generic heatmap with borders. matrix shape: (n_rows x n_cols)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.4), 3))

    ax.imshow(matrix, cmap=cmap, aspect="auto")

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

    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return ax


def plot_onehot(
    sequence: str,
    one_hot: NDArray[np.float32],
    ax=None,
    title="One-Hot Encoded DNA Sequence",
    text=False,
) -> Axes:

    return plot_heatmap(
        matrix=one_hot,
        col_labels=list(sequence),
        row_labels=NUCLEOTIDE_ORDER,
        title=title,
        ax=ax,
        cmap="bwr",
        text=text,
    )


def plot_binary_string(binary: str, ax=None, title="Binary String") -> Axes:
    """Plot a binary string (e.g. '000011110000') as a 1-row heatmap."""
    matrix = np.array([[int(c) for c in binary]])  # shape: (1, len)

    return plot_heatmap(
        matrix=matrix,
        col_labels=list(binary),
        row_labels=[""],
        title=title,
        ax=ax,
        cmap="bwr",
    )


@jx.jaxtyped(typechecker=beartype)
def plot_epistasis_subsetted(
    one_hot: jx.Float[NDArray[np.float32], "sequence_length alphabet_length"],
    hessian_onehot_subsetted: jx.Float[
        NDArray,
        "sequence_length sequence_length",
    ],
    ax: Optional[Axes] = None,
    title="",
):
    if ax is None:
        _, ax = plt.subplots()

    norm = mcolors.TwoSlopeNorm(
        vmin=hessian_onehot_subsetted.min(),
        vcenter=0,
        vmax=hessian_onehot_subsetted.max(),
    )
    im = ax.imshow(hessian_onehot_subsetted, aspect="auto", cmap="bwr", norm=norm)
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xlabel("")

    return ax


def plot_epistasis():
    pass
