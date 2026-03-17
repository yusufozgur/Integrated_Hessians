import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from captum.attr import IntegratedGradients
    import jaxtyping as jx
    from numpy.typing import NDArray
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    return NDArray, Path, jx, mo, np, plt, torch


@app.cell
def _():
    from integrated_hessians.simulation import Nucleotide_Sequence, SimulatedSequence
    from integrated_hessians.simulation.simple_simulation.model import CNNMLP
    from integrated_hessians.simulation.train_model import MotifInteractionsDataset
    from integrated_hessians.simulation.plot import (
        plot_epistasis,
        plot_epistasis_subsetted,
        plot_onehot,
        plot_binary_string,
        plot_heatmap,
    )
    from integrated_hessians import get_hessian
    from integrated_hessians.simulation.test_model import (
        get_test_data,
        get_model,
        plot_training_metrics,
        plot_gif_hessians_from_baseline_to_real,
        get_prediction,
        get_attributions,
        test_and_plot_selected_row,
        interpolate_onehot,
        subset_onehot_hessian,
    )

    return (
        SimulatedSequence,
        get_attributions,
        get_hessian,
        get_model,
        get_prediction,
        get_test_data,
        interpolate_onehot,
        plot_epistasis_subsetted,
        plot_gif_hessians_from_baseline_to_real,
        plot_training_metrics,
        subset_onehot_hessian,
        test_and_plot_selected_row,
    )


@app.cell
def _(mo):
    row = mo.ui.number(0, 1000, 1, label="Choose Row")
    row
    return (row,)


@app.cell
def _(
    NDArray,
    Path,
    SimulatedSequence,
    get_attributions,
    get_hessian,
    get_model,
    get_prediction,
    get_test_data,
    jx,
    np,
    plot_gif_hessians_from_baseline_to_real,
    plot_training_metrics,
    row,
    test_and_plot_selected_row,
    torch,
):

    TEST_DATA = Path("data/simple_simulation/1k_test.json")
    BEST_MODEL = Path("data/simple_simulation/model_best.pth")
    BEST_MODEL_EVAL = Path("data/simple_simulation/model_best_evaluation.json")
    OUTPUT = Path("src/integrated_hessians/simulation/test/")
    SELECTED_ROW = row.value

    test_data = get_test_data(TEST_DATA)
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

    test_row_plot_fig
    # TODO
    # get_integrated_hessian()
    # plot_integrated_hessian()
    return model, one_hot, one_hot_batched


@app.cell
def _(mo):
    baseline_to_input_alpha = mo.ui.slider(
        0, 1, 0.1, show_value=True, label="Baseline to input alpha", value=1
    )
    baseline_to_input_alpha
    return (baseline_to_input_alpha,)


@app.cell
def _(
    baseline_to_input_alpha,
    get_hessian,
    get_prediction,
    interpolate_onehot,
    jx,
    model,
    one_hot: "jx.Float[NDArray[np.float32], \"alphabet_length sequence_length\"]",
    plot_epistasis_subsetted,
    subset_onehot_hessian,
    torch,
):
    interpolation = interpolate_onehot(
        torch.tensor(one_hot), baseline_to_input_alpha.value
    )

    interpolation_hessian: jx.Float[
        torch.Tensor,
        "alphabet_length sequence_length alphabet_length sequence_length",
    ] = get_hessian(
        model=model, input=interpolation.unsqueeze(0).type(torch.float32), target=0
    )

    interpolation_hessian = interpolation_hessian.squeeze(0).squeeze(2)
    interpolation_hessian.shape

    pred_interpolation = get_prediction(model, interpolation.numpy())

    plot_epistasis_subsetted(
        one_hot=one_hot,
        hessian_onehot_subsetted=subset_onehot_hessian(
            calculated_hessian=interpolation_hessian,
            one_hot_mask=torch.tensor(one_hot),
        ).numpy(),
        title=f"Hessian of input with prediction {pred_interpolation: .3f}",
    )
    return (pred_interpolation,)


@app.cell
def _():
    return


@app.cell
def _():
    from integrated_hessians import get_integrated_hessians

    return (get_integrated_hessians,)


@app.cell
def _(mo):
    sampling_steps = mo.ui.number(0, 100, 1, label="Sampling steps", value=1)
    sampling_steps
    return (sampling_steps,)


@app.cell
def _(get_integrated_hessians, model, one_hot_batched, sampling_steps, torch):
    integ_hess_result = get_integrated_hessians(model,one_hot_batched,torch.full_like(one_hot_batched,.25),0, sampling_steps=sampling_steps.value)
    return (integ_hess_result,)


@app.cell
def _(integ_hess_result, plt):
    plt.imshow(integ_hess_result.reshape(200, 200), cmap="bwr")
    return


@app.cell
def _(
    integ_hess_result,
    one_hot: "jx.Float[NDArray[np.float32], \"alphabet_length sequence_length\"]",
    plot_epistasis_subsetted,
    pred_interpolation,
    subset_onehot_hessian,
    torch,
):
    plot_epistasis_subsetted(
        one_hot=one_hot,
        hessian_onehot_subsetted=subset_onehot_hessian(
            calculated_hessian=integ_hess_result.squeeze(0).squeeze(2),
            one_hot_mask=torch.tensor(one_hot),
        ).numpy(),
        title=f"Hessian of input with prediction {pred_interpolation: .3f}",
    )
    return


if __name__ == "__main__":
    app.run()
