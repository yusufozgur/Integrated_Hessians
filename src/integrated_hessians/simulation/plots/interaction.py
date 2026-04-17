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


def plot_interaction():
    pass
