import marimo

__generated_with = "0.23.1"
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
    from matplotlib.figure import Figure
    from sklearn.metrics import r2_score
    import json

    return Figure, NDArray, Path, json, jx, mo, np, plt, r2_score, torch


@app.cell
def _():
    from integrated_hessians.simulation import Nucleotide_Sequence, SimulatedSequence
    from integrated_hessians.simulation.model import CNNMLP
    from integrated_hessians.simulation.train_model import (
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
    from integrated_hessians.simulation.test_model import (
        get_test_data,
        plot_training_metrics,
        plot_gif_hessians_from_baseline_to_real,
        get_prediction,
        get_attributions,
        interpolate_onehot,
        subset_onehot_hessian,
    )

    return (
        CNNMLP,
        SimulatedSequence,
        get_attributions,
        get_hessian,
        get_integrated_hessians,
        get_prediction,
        get_test_data,
        interpolate_onehot,
        plot_binary_string,
        plot_epistasis_subsetted,
        plot_gif_hessians_from_baseline_to_real,
        plot_heatmap,
        plot_onehot,
        plot_training_metrics,
        subset_onehot_hessian,
    )


@app.cell
def _(mo):
    simulation_dropdown = mo.ui.dropdown(
        options=[
            "Simple",
            "Custom Additive and Interaction Effects",
            "Randomized Additive and Interaction Effects",
        ],
        value="Simple",
        label="Choose Simulation",
    )
    simulation_dropdown
    return (simulation_dropdown,)


@app.cell
def _(Path, json, simulation_dropdown):
    config_paths = {
        "Simple": "src/integrated_hessians/simulation/configs/simple.json",
        "Custom Additive and Interaction Effects": "src/integrated_hessians/simulation/configs/custom.json",
        "Randomized Additive and Interaction Effects": "src/integrated_hessians/simulation/configs/random.json",
    }
    chosen_config_path = config_paths[simulation_dropdown.value]
    with open(chosen_config_path,"r") as f:
        config = json.load(f)
    TEST_DATA = Path(config["TEST_DATA"])
    OUT_BEST_MODEL = config["OUT_BEST_MODEL"]
    SEQLEN = config["SEQLEN"]
    TRAIN_DATA = config["TRAIN_DATA"]
    MODEL_WIDTH_MULTIPLIER = config["MODEL_WIDTH_MULTIPLIER"]
    return MODEL_WIDTH_MULTIPLIER, OUT_BEST_MODEL, SEQLEN, TEST_DATA


@app.cell
def _(TEST_DATA):
    TEST_DATA
    return


@app.cell
def _(SEQLEN, TEST_DATA, get_test_data):
    test_data = get_test_data(TEST_DATA, SEQLEN)
    return (test_data,)


@app.cell
def _(test_data):
    set_of_names = sorted(list(set([x.motif_names[0] for x in test_data])))
    set_of_names
    return (set_of_names,)


@app.cell
def _(mo, set_of_names):
    motif1_choose = mo.ui.dropdown(
        options=set_of_names + ["Any"], value="Any", label="Motif1:"
    )
    motif2_choose = mo.ui.dropdown(
        options=set_of_names + ["Any"], value="Any", label="Motif2:"
    )
    motif1_choose, motif2_choose
    return motif1_choose, motif2_choose


@app.cell
def _(motif1_choose, motif2_choose, test_data):
    subsetted_data = test_data
    if motif1_choose.value is not "Any":
        subsetted_data = [
            x for x in subsetted_data if x.motif_names[0] == motif1_choose.value
        ]
    if motif2_choose.value is not "Any":
        subsetted_data = [
            x for x in subsetted_data if x.motif_names[1] == motif2_choose.value
        ]
    return (subsetted_data,)


@app.cell
def _(all_phens, all_preds, plt, r2_score):
    plt.scatter(all_phens, all_preds, alpha=0.01)
    r2 = r2_score(all_phens, all_preds)
    plt.title(f"phens vs preds. R2: {r2}")
    return


@app.cell
def _(mo):
    SELECTEd_ROW = mo.ui.number(0, 200, 1, label="Choose Row")
    SELECTEd_ROW
    return (SELECTEd_ROW,)


@app.cell
def _(
    CNNMLP,
    Figure,
    MODEL_WIDTH_MULTIPLIER,
    NDArray,
    OUT_BEST_MODEL,
    SELECTEd_ROW,
    SEQLEN,
    SimulatedSequence,
    get_attributions,
    get_hessian,
    get_prediction,
    jx,
    np,
    plot_binary_string,
    plot_gif_hessians_from_baseline_to_real,
    plot_heatmap,
    plot_onehot,
    plot_training_metrics,
    plt,
    subsetted_data,
    torch,
):
    test_row: SimulatedSequence = subsetted_data[SELECTEd_ROW.value]
    model = CNNMLP(sequence_length=SEQLEN, width_multiplier=MODEL_WIDTH_MULTIPLIER)
    model.load_state_dict(torch.load(OUT_BEST_MODEL))
    model.eval()
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

    # test_row_plot_fig, _ = test_and_plot_selected_row(
    #     sequence=test_row.nucleotides,
    #     one_hot=one_hot_permuted,
    #     attributions=attributions_permuted,
    #     integrated_gradients_delta=float(ig_delta),
    #     real_attributions=real_attributions,
    #     phenotype=test_row.phenotype,
    #     prediction=row_prediction,
    #     motif_mask_1=test_row.motif_mask_1,
    #     motif_type_1=test_row.motif_types[0].name,
    #     motif_mask_2=test_row.motif_mask_2,
    #     motif_type_2=test_row.motif_types[1].name,
    #     calculated_hessian=calculated_hessian,
    # )

    test_row_plot_fig: Figure
    test_row_plot_axes: np.ndarray
    test_row_plot_fig, test_row_plot_axes = plt.subplots(
        ncols=1,
        nrows=5,
        figsize=(10, 6),
        sharex=True,  # guarantees column alignment
        height_ratios=[8, 1, 1, 8, 1],
        layout="constrained",
    )

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

    test_row_plot_fig
    return model, one_hot, one_hot_batched, real_attributions, test_row


@app.cell
def _(model, test_data, torch):
    onehot_all = torch.tensor([x.one_hot for x in test_data], dtype=torch.float)
    all_phens = [x.phenotype for x in test_data]
    with torch.no_grad():
        all_preds = model(onehot_all)
    return all_phens, all_preds


@app.cell
def _(np, real_attributions, test_row: "SimulatedSequence"):
    mask1 = np.array([int(x) for x in test_row.motif_mask_1])
    mask2 = np.array([int(x) for x in test_row.motif_mask_2])
    (
        f"First motif with name: {test_row.motif_names[0]} has attr sum: {(real_attributions[0] * mask1).sum(): .3f}",
        f"Second motif with name: {test_row.motif_names[1]} has attr sum: {(real_attributions[0] * mask2).sum(): .3f}",
    )
    return mask1, mask2


@app.cell
def _(mo):
    show_hessian = mo.ui.checkbox(label="Show Hessian")
    show_integrated_hessian = mo.ui.checkbox(
        label="Show Integrated Hessian (Our implementation)"
    )
    show_integrated_hessian_janizeketal = mo.ui.checkbox(
        label="Show Integrated Hessian (Janizek Implementation)"
    )
    mo.vstack(
        [show_hessian, show_integrated_hessian, show_integrated_hessian_janizeketal]
    )
    return (
        show_hessian,
        show_integrated_hessian,
        show_integrated_hessian_janizeketal,
    )


@app.cell
def _(mo, show_hessian):
    baseline_to_input_alpha = mo.ui.slider(
        0, 1, 0.1, show_value=True, label="Baseline to input alpha", value=1
    )
    baseline_to_input_alpha if show_hessian.value else None
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
    show_hessian,
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

    hessian_interaction_plot = plot_epistasis_subsetted(
        hessian_onehot_subsetted=subset_onehot_hessian(
            calculated_hessian=interpolation_hessian,
            one_hot_mask=torch.tensor(one_hot),
        ).numpy(),
        title=f"Hessian of input with prediction {pred_interpolation: .3f}",
    )
    hessian_interaction_plot if show_hessian.value else None
    return


@app.cell
def _(mo, show_integrated_hessian):
    sampling_steps = mo.ui.number(0, 200, 1, label="Sampling steps", value=10)
    mo.vstack(
        [
            "Increasing the sampling steps will make integrated hessians approximation more accurate",
            sampling_steps,
        ]
    ) if show_integrated_hessian.value else None
    return (sampling_steps,)


@app.cell
def _(
    get_integrated_hessians,
    model,
    one_hot: "jx.Float[NDArray[np.float32], \"alphabet_length sequence_length\"]",
    one_hot_batched,
    plot_epistasis_subsetted,
    plt,
    sampling_steps,
    show_integrated_hessian,
    subset_onehot_hessian,
    torch,
):
    if show_integrated_hessian.value:
        integ_hess_result, ih_delta = get_integrated_hessians(
            model=model,
            inputs=one_hot_batched,
            baselines=torch.full_like(one_hot_batched, 0.25),
            target=0,
            approximation_steps=sampling_steps.value,
            optimize_for_duplicate_interpolation_values=True,
        )

        integ_hess_plots_fig, integ_hess_plots_ax = plt.subplots(
            nrows=1, ncols=2, figsize=(10, 4)
        )

        ih_interaction_plot_subsetted = plot_epistasis_subsetted(
            hessian_onehot_subsetted=subset_onehot_hessian(
                calculated_hessian=integ_hess_result.squeeze(0),
                one_hot_mask=torch.tensor(one_hot),
            )
            .detach()
            .numpy(),
            title=f"Integrated hessians. delta: {ih_delta[0]: .3f}",
            ax=integ_hess_plots_ax[0],
        )
        integ_hess_plots_ax[1].imshow(
            integ_hess_result.detach().reshape(400, 400), cmap="bwr"
        )
    else:
        integ_hess_plots_fig = None

    integ_hess_plots_fig
    return (integ_hess_result,)


@app.cell
def _(
    all_sum,
    ihnp_masked_selfinteractmotif1,
    ihnp_masked_selfinteractmotif2,
    ihnp_masked_sum_pair1,
    ihnp_masked_sum_pair2,
    integ_hess_result,
    show_integrated_hessian_janizeketal,
):
    (
        f"{integ_hess_result.sum() = :.3f}",
        f"{ihnp_masked_sum_pair1 = :.3f}",
        f"{ihnp_masked_sum_pair2 = :.3f}",
        f"{ihnp_masked_sum_pair1 + ihnp_masked_sum_pair2 = :.3f}",
        f"{ihnp_masked_selfinteractmotif1 = :.3f}",
        f"{ihnp_masked_selfinteractmotif2 = :.3f}",
        f"{all_sum = :.3f}",
    ) if show_integrated_hessian_janizeketal.value else None
    return


@app.cell
def _(
    SEQLEN,
    integ_hess_result,
    mask1,
    mask2,
    one_hot: "jx.Float[NDArray[np.float32], \"alphabet_length sequence_length\"]",
    show_integrated_hessian,
    subset_onehot_hessian,
    torch,
):
    if show_integrated_hessian.value:
        ihnp = integ_hess_result.squeeze(0).numpy()
        ihrealnp = (
            subset_onehot_hessian(
                calculated_hessian=integ_hess_result.squeeze(0),
                one_hot_mask=torch.tensor(one_hot),
            )
        ).numpy()
    
        ihrealnp_masked_sum_pair1 = (
            ihrealnp * mask1.reshape(SEQLEN, 1) * mask2.reshape(1, SEQLEN)
        ).sum()
        ihrealnp_masked_sum_pair2 = (
            ihrealnp * mask2.reshape(SEQLEN, 1) * mask1.reshape(1, SEQLEN)
        ).sum()
    
        ihnp_masked_sum_pair1 = (
            ihnp * mask1.reshape(SEQLEN, 1, 1, 1) * mask2.reshape(1, 1, SEQLEN, 1)
        ).sum()
        ihnp_masked_sum_pair2 = (
            ihnp * mask1.reshape(1, 1, SEQLEN, 1) * mask2.reshape(SEQLEN, 1, 1, 1)
        ).sum()
    
        ihrealnp_masked_selfinteractmotif1 = (
            ihrealnp * mask1.reshape(SEQLEN, 1) * mask1.reshape(1, SEQLEN)
        ).sum()
    
        ihrealnp_masked_selfinteractmotif2 = (
            ihrealnp * mask2.reshape(SEQLEN, 1) * mask2.reshape(1, SEQLEN)
        ).sum()
    
        ihnp_masked_selfinteractmotif1 = (
            ihnp * mask1.reshape(1, 1, SEQLEN, 1) * mask1.reshape(SEQLEN, 1, 1, 1)
        ).sum()
    
        ihnp_masked_selfinteractmotif2 = (
            ihnp * mask2.reshape(1, 1, SEQLEN, 1) * mask2.reshape(SEQLEN, 1, 1, 1)
        ).sum()
    
        allrealsum = (
            ihrealnp_masked_sum_pair1
            + ihrealnp_masked_sum_pair2
            + ihrealnp_masked_selfinteractmotif1
            + ihrealnp_masked_selfinteractmotif2
        )
    
        all_sum = (
            ihnp_masked_sum_pair1
            + ihnp_masked_sum_pair2
            + ihnp_masked_selfinteractmotif1
            + ihnp_masked_selfinteractmotif2
        )

    (
        f"{ihrealnp_masked_sum_pair1 = :.3f}",
        f"{ihrealnp_masked_sum_pair2 = :.3f}",
        f"{ihrealnp_masked_sum_pair1 + ihrealnp_masked_sum_pair2 = :.3f}",
        f"{ihrealnp_masked_selfinteractmotif1 = :.3f}",
        f"{ihrealnp_masked_selfinteractmotif2 = :.3f}",
        f"{allrealsum = :.3f}",
        f"{ihrealnp.sum() = :.3f}",
        f"{integ_hess_result.sum() = :.3f}",
        f"{ihnp_masked_sum_pair1 = :.3f}",
        f"{ihnp_masked_sum_pair2 = :.3f}",
        f"{ihnp_masked_sum_pair1 + ihnp_masked_sum_pair2 = :.3f}",
        f"{ihnp_masked_selfinteractmotif1 = :.3f}",
        f"{ihnp_masked_selfinteractmotif2 = :.3f}",
        f"{all_sum = :.3f}",
    ) if show_integrated_hessian.value else None
    return (
        all_sum,
        ihnp_masked_selfinteractmotif1,
        ihnp_masked_selfinteractmotif2,
        ihnp_masked_sum_pair1,
        ihnp_masked_sum_pair2,
    )


@app.cell
def _(mo, show_integrated_hessian_janizeketal):
    sampling_steps_janizeketal = mo.ui.number(
        0, 200, 1, label="Sampling steps (janizek et al)", value=10
    )
    mo.vstack(
        [
            "Path explain implementation takes square root of the num_samples variable, so if num of steps is 60 for us, it will be 3600 for them.",
            sampling_steps_janizeketal,
        ]
    ) if show_integrated_hessian_janizeketal.value else None
    return (sampling_steps_janizeketal,)


@app.cell
def _(
    SEQLEN,
    model,
    one_hot: "jx.Float[NDArray[np.float32], \"alphabet_length sequence_length\"]",
    one_hot_batched,
    plot_epistasis_subsetted,
    sampling_steps_janizeketal,
    show_integrated_hessian_janizeketal,
    subset_onehot_hessian,
    torch,
):
    from path_explain import PathExplainerTorch

    if show_integrated_hessian_janizeketal.value:

        def exp_reshaper(x: torch.Tensor):
            x = x.reshape((1, SEQLEN, 4))
            x = model(x)
            return x

        exp = PathExplainerTorch(exp_reshaper)

        exp_input = one_hot_batched.reshape(1, SEQLEN * 4)
        exp_baseline = torch.full_like(exp_input, 0.25)
        exp_baseline.shape

        exp_input.requires_grad_(True)
        exp_baseline.requires_grad_(True)

        exp_ih = exp.interactions(
            exp_input,
            exp_baseline,
            num_samples=sampling_steps_janizeketal.value,
            use_expectation=False,
        )

        exp_ih_delta = (
            model(one_hot_batched)
            - model(torch.full_like(one_hot_batched, 0.25))
            - exp_ih.sum()
        )

        exp_ih_reshaped = exp_ih.reshape(1, SEQLEN, 4, SEQLEN, 4)
        exp_ih_reshaped.shape

        exp_ih_interaction_plot = plot_epistasis_subsetted(
            hessian_onehot_subsetted=subset_onehot_hessian(
                calculated_hessian=exp_ih_reshaped.detach().squeeze(0),
                one_hot_mask=torch.tensor(one_hot),
            ).numpy(),
            title=f"Integrated hessians via path explain. delta: {exp_ih_delta}",
        )
    else:
        exp_ih_interaction_plot = None
    exp_ih_interaction_plot
    # plt.imshow(exp_ih_reshaped.detach().numpy().reshape(SEQLEN*4, SEQLEN*4), cmap="bwr")
    return


if __name__ == "__main__":
    app.run()
