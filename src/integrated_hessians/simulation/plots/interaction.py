from typing import Optional
from beartype import beartype
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import matplotlib.colors as mcolors
import jaxtyping as jx


@jx.jaxtyped(typechecker=beartype)
def plot_interaction_subsetted(
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

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    ax.set_xlabel("")

    return ax


NUCLEOTIDES = ["A", "C", "G", "T"]


def plot_genomic_interaction(
    hessian_onehot: jx.Float[
        NDArray,
        "sequence_length sequence_length 4 4",
    ],
    ax: Optional[Axes] = None,
    fig_width=6,
    fig_height=4,
    title="",
    cmap="RdBu_r",
    show_nuc_labels=True,
    show_grid_lines=True,
    colorbar=True,
):
    """
    Plot a genomic interaction heatmap from a Hessian matrix.

    Each cell (i, j) in the sequence_length × sequence_length grid contains
    a 4×4 sub-grid showing interactions between all nucleotide pairs at
    positions i and j.

    Parameters
    ----------
    hessian_onehot : ndarray of shape (L, L, 4, 4)
        Hessian matrix in one-hot nucleotide basis.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if None.
    fig_width, fig_height : float
        Figure dimensions in inches.
    title : str
        Plot title.
    cmap : str
        Colormap name (diverging recommended).
    show_nuc_labels : bool
        Whether to draw A/C/G/T tick labels inside each cell.
    show_grid_lines : bool
        Whether to draw separator lines between sequence positions.
    colorbar : bool
        Whether to add a colorbar.
    """
    seq_len = hessian_onehot.shape[0]
    assert hessian_onehot.shape == (seq_len, seq_len, 4, 4), (
        f"Expected shape (L, L, 4, 4), got {hessian_onehot.shape}"
    )

    # ------------------------------------------------------------------ #
    # Build the flat (seq_len*4) × (seq_len*4) display matrix             #
    # ------------------------------------------------------------------ #
    n = seq_len * 4
    flat = np.zeros((n, n))
    for i in range(seq_len):
        for j in range(seq_len):
            flat[i * 4 : i * 4 + 4, j * 4 : j * 4 + 4] = hessian_onehot[i, j]

    # ------------------------------------------------------------------ #
    # Axes / figure setup                                                  #
    # ------------------------------------------------------------------ #
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.get_figure()

    # Diverging norm centred at 0
    abs_max = np.nanmax(np.abs(flat))
    if abs_max == 0:
        abs_max = 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    im = ax.imshow(flat, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")

    # ------------------------------------------------------------------ #
    # Tick marks: one tick per nucleotide sub-cell                         #
    # ------------------------------------------------------------------ #
    tick_positions = np.arange(n)  # 0 … seq_len*4 - 1

    if show_nuc_labels:
        nuc_labels = NUCLEOTIDES * seq_len
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(nuc_labels, fontsize=4, rotation=90)
        ax.set_yticklabels(nuc_labels, fontsize=4)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # ------------------------------------------------------------------ #
    # Sequence-position labels centred above / left of each 4-cell block  #
    # ------------------------------------------------------------------ #
    pos_ticks = [i * 4 + 1.5 for i in range(seq_len)]
    # Use a secondary axis for cleaner position labels
    ax2x = ax.secondary_xaxis("top")
    ax2x.set_xticks(pos_ticks)
    ax2x.set_xticklabels([str(i) for i in range(seq_len)], fontsize=6)
    ax2x.tick_params(length=0)

    ax2y = ax.secondary_yaxis("right")
    ax2y.set_yticks(pos_ticks)
    ax2y.set_yticklabels([str(i) for i in range(seq_len)], fontsize=6)
    ax2y.tick_params(length=0)

    # ------------------------------------------------------------------ #
    # Grid lines between sequence positions                                #
    # ------------------------------------------------------------------ #
    if show_grid_lines:
        for k in range(1, seq_len):
            coord = k * 4 - 0.5
            ax.axhline(coord, color="black", linewidth=0.8, alpha=0.6)
            ax.axvline(coord, color="black", linewidth=0.8, alpha=0.6)

    # ------------------------------------------------------------------ #
    # Faint lines between nucleotide sub-cells                             #
    # ------------------------------------------------------------------ #
    for k in range(n):
        if k % 4 != 0:  # skip major lines already drawn above
            ax.axhline(k - 0.5, color="grey", linewidth=0.2, alpha=0.3)
            ax.axvline(k - 0.5, color="grey", linewidth=0.2, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Colorbar & title                                                     #
    # ------------------------------------------------------------------ #
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Interaction strength", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    ax.set_title(title, fontsize=9, pad=14)
    ax.set_xlabel("Position j", fontsize=7)
    ax.set_ylabel("Position i", fontsize=7)

    fig.tight_layout()
    return ax
