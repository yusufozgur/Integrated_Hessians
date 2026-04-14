from pathlib import Path
from captum.attr import IntegratedGradients
import jaxtyping as jx
from numpy.typing import NDArray
import torch
import numpy as np

from integrated_hessians.simulation import SimulatedSequence
from integrated_hessians.simulation.model import CNNMLP
from integrated_hessians.simulation.train_model import (
    MotifInteractionsDataset,
)
from beartype import beartype


def get_test_data(test_path: Path, SEQLEN: int) -> list[SimulatedSequence]:
    assert test_path.exists()
    assert test_path.is_file()
    seqs = MotifInteractionsDataset(input=test_path, SEQLEN=SEQLEN).data
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
