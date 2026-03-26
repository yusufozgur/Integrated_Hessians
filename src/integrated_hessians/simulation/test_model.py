from pathlib import Path
from captum.attr import IntegratedGradients
import jaxtyping as jx
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.typing import NDArray
import torch
import numpy as np

from integrated_hessians.simulation import Nucleotide_Sequence, SimulatedSequence
from integrated_hessians.simulation.simple_simulation.model import CNNMLP
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
from integrated_hessians import get_hessian
from beartype import beartype

TEST_DATA = Path("data/simple_simulation/1k_test.json")
BEST_MODEL = Path("data/simple_simulation/model_best.pth")
BEST_MODEL_EVAL = Path("data/simple_simulation/model_best_evaluation.json")
OUTPUT = Path("src/integrated_hessians/simulation/test/")


def main():
    test_data = get_test_data(TEST_DATA)
    for SELECTED_ROW in range(5):
        test_row: SimulatedSequence = test_data[SELECTED_ROW]
        model = get_model(BEST_MODEL)
        # TODO
        plot_training_metrics()
        plot_gif_hessians_from_baseline_to_real()
        one_hot: jx.Float[NDArray[np.float32], "alphabet_length sequence_length"] = (
            test_row.one_hot
        )
        row_prediction = get_prediction(model=model, one_hot=one_hot)
        attributions, ig_delta = get_attributions(model=model, one_hot=one_hot)
        one_hot_permuted: jx.Float[
            NDArray[np.float32], "alphabet_length sequence_length"
        ] = one_hot.transpose((1, 0))
        attributions_permuted: jx.Float[
            NDArray[np.float32], "alphabet_length sequence_length"
        ] = attributions.squeeze(0).numpy().transpose((1, 0))
        real_attributions = np.sum(
            one_hot_permuted * attributions_permuted, axis=0
        ).reshape(1, -1)
        one_hot_batched = torch.tensor(one_hot).type(torch.float32).unsqueeze(0)
        calculated_hessian: jx.Float[
            torch.Tensor,
            "batch_size alphabet_length sequence_length batch_size alphabet_length sequence_length",
        ] = get_hessian(model=model, input=one_hot_batched, target=0)
        # batch size is 1, so remove that dimension
        calculated_hessian: jx.Float[
            torch.Tensor,
            "alphabet_length sequence_length alphabet_length sequence_length",
        ] = calculated_hessian.squeeze(0).squeeze(2)
        test_row_plot_fig, _ = test_and_plot_selected_row(
            sequence=test_row.nucleotides,
            one_hot=one_hot_permuted,
            integrated_gradients_delta=float(ig_delta),
            attributions=attributions_permuted,
            real_attributions=real_attributions,
            phenotype=test_row.phenotype,
            prediction=row_prediction,
            motif_mask_1=test_row.motif_mask_1,
            motif_type_1=test_row.motif_types[0].name,
            motif_mask_2=test_row.motif_mask_2,
            motif_type_2=test_row.motif_types[1].name,
            calculated_hessian=calculated_hessian,
        )
        test_row_plot_fig.savefig(OUTPUT / f"test_row_{SELECTED_ROW}.svg")
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
def test_and_plot_selected_row(
    sequence: Nucleotide_Sequence,
    one_hot: jx.Float[NDArray[np.float32], "alphabet_length sequence_length"],
    attributions: jx.Float[NDArray[np.float32], "alphabet_length sequence_length"],
    integrated_gradients_delta: float,
    real_attributions: jx.Float[NDArray[np.float32], "1 sequence_length"],
    phenotype: float,
    prediction: float,
    motif_mask_1: str,
    motif_type_1: str,
    motif_mask_2: str,
    motif_type_2: str,
    calculated_hessian: jx.Float[
        torch.Tensor,
        "sequence_length alphabet_length sequence_length alphabet_length",
    ],
) -> tuple[Figure, ndarray]:
    """
    Plot, from top down
        -   Sequence Logo
        -   Onehot Sequence Heatmap
        -   Annotation of motif 1 and 2 positions in sequence, alongside the motif roles
        -   Integrated Gradients
        -   Epistasis Map (For now heatmap, future: rotated half triangle)
            -   of hessian of input
            -   of integrated hessian of input
    """
    fig: Figure
    axes: ndarray
    fig, axes = plt.subplots(
        ncols=1,
        nrows=5,
        figsize=(10, 6),
        sharex=True,  # guarantees column alignment
        height_ratios=[4, 1, 1, 4, 1],
        layout="constrained",
    )

    plot_onehot(
        sequence=sequence,
        one_hot=one_hot,
        ax=axes[0],
        title=f"Phen: {phenotype} Pred: {prediction: .3}",
    )
    plot_binary_string(motif_mask_1, axes[1], title=motif_type_1)
    plot_binary_string(motif_mask_2, axes[2], title=motif_type_2)
    plot_onehot(
        sequence=sequence,
        one_hot=attributions,
        ax=axes[3],
        title=f"Integrated Gradients (Multiplied input: true), delta: {integrated_gradients_delta: .3f}",
    )
    plot_heatmap(
        matrix=real_attributions,
        row_labels=["Real base"],
        col_labels=list(sequence),
        ax=axes[4],
        cmap="bwr",
        title="Real Attributions",
    )

    # plt.tight_layout() # alternative to layout='constrained'

    return fig, axes


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
