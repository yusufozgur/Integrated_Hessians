import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from integrated_hessians.simulation import NUCLEOTIDE_ORDER
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_heatmap(matrix: np.ndarray, col_labels: list, row_labels: list, title: str, ax=None, cmap='Blues', text = False, add_colorbar=True):
    """Generic heatmap with borders. matrix shape: (n_rows x n_cols)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.4), 3))

    ax.imshow(matrix, cmap=cmap, aspect='auto')

    n_rows, n_cols = matrix.shape
    for x in range(n_cols + 1):
        ax.axvline(x - 0.5, color='black', linewidth=1.5)
    for y in range(n_rows + 1):
        ax.axhline(y - 0.5, color='black', linewidth=1.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)

    if text:
        for i in range(n_rows):
            for j in range(n_cols):
                ax.text(j, i, f'{matrix[i, j]:.0f}', ha='center', va='center', color='gray', fontsize=5)

    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Base")


    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return ax


def plot_onehot(sequence: str, one_hot: NDArray, ax=None, title="One-Hot Encoded DNA Sequence", text = False):

    return plot_heatmap(
        matrix=one_hot,
        col_labels=list(sequence),
        row_labels=NUCLEOTIDE_ORDER,
        title=title,
        ax=ax,
        cmap='Blues',
        text=text
    )


def plot_binary_string(binary: str, ax=None, title="Binary String"):
    """Plot a binary string (e.g. '000011110000') as a 1-row heatmap."""
    matrix = np.array([[int(c) for c in binary]])  # shape: (1, len)

    return plot_heatmap(
        matrix=matrix,
        col_labels=list(binary),
        row_labels=[''],
        title=title,
        ax=ax,
        cmap='Blues'
    )
