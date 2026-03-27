from pathlib import Path
from captum.attr import IntegratedGradients
import jaxtyping as jx
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.typing import NDArray
import torch
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

from integrated_hessians.simulation import Nucleotide_Sequence, SimulatedSequence
from integrated_hessians.simulation.model import CNNMLP
from integrated_hessians.simulation.simple_simulation.train_model import (
    MotifInteractionsDataset,
)
from integrated_hessians.simulation.plot import (
    plot_epistasis,
    plot_epistasis_subsetted,
    plot_onehot,
    plot_binary_string,
    plot_heatmap,
)
from integrated_hessians import get_hessian, get_integrated_hessians
from beartype import beartype

from integrated_hessians.simulation.custom_additive_and_interactive_effects.config import (
    TEST_DATA,
    OUT_BEST_MODEL,
    OUT_BEST_MODEL_EVAL,
    TEST_OUTPUT,
    INTEGRATED_HESSIANS_SAMPLING_STEPS,
    NUM_OF_ROWS_TESTED,
)


def main():
    test_data = get_test_data(TEST_DATA)

    # If we plot first n rows, it is not guaranteed we will see interesting rows
    # so instead we will plot first n interesting rows per possible phenotype value
    # half interaction half additive: 6, 6,1

    interesting_pairs = [
        (("Motif1", "Motif2"), "half interaction"),
        (("Motif2", "Motif3"), "full interaction"),
        (("Motif4", "Random1"), "half additive"),
        (("Motif4", "Motif4"), "half additive twice"),
        (("Motif5", "Random1"), "full additive"),
        (("Motif5", "Motif5"), "full additive twice"),
        (("Motif6", "Random1"), "half interaction half additive"),
        (("Motif6", "Motif1"), "half interaction half additive"),
        (("Random2", "Motif1"), "random"),
    ]
    interesting_rows: list[SimulatedSequence] = []
    labels = []

    for pair, label in interesting_pairs:
        add_rows = [row for row in test_data if row.motif_names == pair][
            :NUM_OF_ROWS_TESTED
        ]
        assert len(add_rows) == NUM_OF_ROWS_TESTED
        interesting_rows += add_rows
        labels = [
            *labels,
            *(
                [
                    label,
                ]
                * NUM_OF_ROWS_TESTED
            ),
        ]

    for i, (label, test_row) in tqdm(enumerate(tzip(labels, interesting_rows))):
        # Plot interesting graphs for SELECTED_ROW
        # - Logo sequence
        # - One hot encoded sequence heatmap
        # - Annotate motif 1 and motif 2 location in heatmap, label their names/roles
        # - Plot Integrated Gradients heatmap
        # - Subset integrated gradients for existing nucleotides and show in heatmap
        # - show hessian for input
        # - show integrated hessian for input and baseline

        # === PREPARE DATA ===
        model = get_model(Path(OUT_BEST_MODEL))

        one_hot: jx.Float[NDArray[np.float32], "alphabet_length sequence_length"] = (
            test_row.one_hot
        )
        one_hot_permuted: jx.Float[
            NDArray[np.float32], "alphabet_length sequence_length"
        ] = one_hot.transpose((1, 0))
        one_hot_batched: jx.Float[torch.Tensor, "1 alphabet_length sequence_length"] = (
            torch.tensor(one_hot).type(torch.float32).unsqueeze(0)
        )

        row_prediction = get_prediction(model=model, one_hot=one_hot)

        attributions, ig_delta = get_attributions(model=model, one_hot=one_hot)
        attributions_permuted: jx.Float[
            NDArray[np.float32], "alphabet_length sequence_length"
        ] = attributions.squeeze(0).numpy().transpose((1, 0))
        real_attributions = np.sum(
            one_hot_permuted * attributions_permuted, axis=0
        ).reshape(1, -1)

        # === PLOT ===

        # TODO
        plot_training_metrics()
        # plot predicted vs actual phenotype correlation
        plot_gif_hessians_from_baseline_to_real()

        test_row_plot_fig: Figure
        test_row_plot_axes: ndarray
        test_row_plot_fig, test_row_plot_axes = plt.subplots(
            ncols=1,
            nrows=6,
            figsize=(10, 16),
            sharex=True,  # guarantees column alignment
            height_ratios=[8, 1, 1, 8, 1, 40],
            layout="constrained",
        )

        # - Logo sequence
        # TODO
        # - One hot encoded sequence heatmap
        plot_onehot(
            sequence=test_row.nucleotides,
            one_hot=one_hot_permuted,
            ax=test_row_plot_axes[0],
            title=f"Phen: {test_row.phenotype} Pred: {row_prediction: .3}",
        )
        # - Annotate motif 1 and motif 2 location in heatmap, label their names/roles
        plot_binary_string(
            test_row.motif_mask_1,
            test_row_plot_axes[1],
            title=test_row.motif_names[0],
        )
        plot_binary_string(
            test_row.motif_mask_2,
            test_row_plot_axes[2],
            title=test_row.motif_names[1],
        )
        # - Plot Integrated Gradients heatmap
        plot_onehot(
            sequence=test_row.nucleotides,
            one_hot=attributions_permuted,
            ax=test_row_plot_axes[3],
            title=f"Integrated Gradients (Multiplied input: true), delta: {float(ig_delta): .3f}",
            cmap="bwr",
        )
        # - Subset integrated gradients for existing nucleotides and show in heatmap
        plot_heatmap(
            matrix=real_attributions,
            row_labels=["Real base"],
            col_labels=list(test_row.nucleotides),
            ax=test_row_plot_axes[4],
            cmap="bwr",
            title="Real Attributions",
        )

        # - show hessian for input

        # choose if you wanna force it to be a square, but then it wont share the x axis
        # test_row_plot_axes[5].set_box_aspect(1)

        # OPTIONAL, UNCOMMENT IF YOU WANNA PLOT
        """
        hessian_result: jx.Float[
            torch.Tensor,
            "alphabet_length sequence_length alphabet_length sequence_length",
        ] = (
            get_hessian(model=model, input=one_hot_batched, target=0)
            .squeeze(0)
            .squeeze(2)
        )

        plot_epistasis_subsetted(
            hessian_onehot_subsetted=subset_onehot_hessian(
                calculated_hessian=hessian_result,
                one_hot_mask=torch.tensor(one_hot),
            ).numpy(),
            title=f"Hessian of input with prediction {row_prediction: .3f}",
            ax=test_row_plot_axes[5],
        )
        """

        # - show integrated hessian for input and baseline

        integ_hess_result, ih_delta = get_integrated_hessians(
            model=model,
            inputs=one_hot_batched,
            baselines=torch.full_like(one_hot_batched, 0.0),
            target=0,
            approximation_steps=INTEGRATED_HESSIANS_SAMPLING_STEPS,
            optimize_for_duplicate_interpolation_values=True,
        )

        plot_epistasis_subsetted(
            hessian_onehot_subsetted=subset_onehot_hessian(
                calculated_hessian=integ_hess_result.squeeze(0),
                one_hot_mask=torch.tensor(one_hot),
            )
            .detach()
            .numpy(),
            title=f"Integrated hessians. Subsetted. delta: {ih_delta[0]: .3f}",
            ax=test_row_plot_axes[5],
        )

        # plt.tight_layout() # alternative to layout='constrained'

        test_row_plot_fig.savefig(TEST_OUTPUT / f"test_row_{i}_{label}.svg")
        # TODO
        # get_integrated_hessian()
        # plot_integrated_hessian()


def get_test_data(test_path: Path) -> list[SimulatedSequence]:
    assert test_path.exists()
    assert test_path.is_file()
    seqs = MotifInteractionsDataset(test_path).data
    return seqs


def get_model(best_model: Path) -> torch.nn.Module:
    model = CNNMLP()
    model.load_state_dict(torch.load(best_model))
    model.eval()
    return model


@jx.jaxtyped(typechecker=beartype)
def get_prediction(
    model: torch.nn.Module,
    one_hot: jx.Float[NDArray[np.float32], "sequence_length alphabet_length"],
) -> float:
    batched_input = torch.tensor(one_hot).unsqueeze(0)
    with torch.no_grad():
        pred = model(batched_input.type(torch.float))
    pred = float(pred[0, 0])
    return pred


@jx.jaxtyped(typechecker=beartype)
def get_attributions(
    model: torch.nn.Module,
    one_hot: jx.Float[NDArray[np.float32], "alphabet_length sequence_length"],
):

    input = torch.tensor(one_hot).unsqueeze(0).type(torch.float)
    baseline = torch.full_like(input, 0)
    ig = IntegratedGradients(model, multiply_by_inputs=True)
    attributions, delta = ig.attribute(input, baseline, return_convergence_delta=True)
    return attributions, delta


def interpolate_onehot(onehot_tensor, alpha):
    uniform = torch.full_like(onehot_tensor, 0.25)
    return (1 - alpha) * uniform + alpha * onehot_tensor


def plot_training_metrics():
    """
    Plot test and validation losses, R2 vs epoch. Also plot phenotype vs predictions on test set alongisde R2.
    """
    pass


@jx.jaxtyped(typechecker=beartype)
def subset_onehot_hessian(
    calculated_hessian: jx.Float[
        torch.Tensor,
        "sequence_length alphabet_length sequence_length alphabet_length",
    ],
    one_hot_mask: jx.Float[
        torch.Tensor,
        "sequence_length alphabet_length",
    ],
) -> jx.Float[
    torch.Tensor,
    "sequence_length sequence_length",
]:
    second_order_onehot: jx.Float[
        torch.Tensor,
        "sequence_length alphabet_length sequence_length alphabet_length",
    ] = one_hot_mask.reshape(1, 1, *one_hot_mask.shape) * one_hot_mask.reshape(
        *one_hot_mask.shape, 1, 1
    )

    subsetted: jx.Float[
        torch.Tensor,
        "sequence_length sequence_length",
    ] = (calculated_hessian * second_order_onehot).sum(1).sum(2)

    return subsetted


def plot_gif_hessians_from_baseline_to_real():
    pass


if __name__ == "__main__":
    main()
