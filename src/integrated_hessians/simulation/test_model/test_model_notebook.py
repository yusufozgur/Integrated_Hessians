import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    config_paths = {
        "Simple": "src/integrated_hessians/simulation/configs/simple.json",
        "Custom Additive and Interaction Effects": "src/integrated_hessians/simulation/configs/custom.json",
        "Randomized Additive and Interaction Effects": "src/integrated_hessians/simulation/configs/random.json",
    }
    evaluation_paths = {
        "Simple": "data/simple_simulation/model_best_evaluation.json",
        "Custom Additive and Interaction Effects": "data/custom_additive_and_interactive_effects/model_best_evaluation.json",
        "Randomized Additive and Interaction Effects": "data/randomized_additive_and_interactive_effects/model_best_evaluation.json",
    }
    figs_common_width = 14
    fig_genomic_line_height = .125
    return (
        config_paths,
        evaluation_paths,
        fig_genomic_line_height,
        figs_common_width,
    )


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
    from torch import Tensor

    return Path, Tensor, json, jx, mo, np, plt, r2_score, torch


@app.cell
def _():
    from integrated_hessians.simulation import Nucleotide_Sequence, SimulatedSequence
    from integrated_hessians.simulation.model import CNNMLP
    from integrated_hessians.simulation.train_model import (
        MotifInteractionsDataset,
    )
    from integrated_hessians.simulation import NUCLEOTIDE_ORDER
    from integrated_hessians.simulation.plots.heatmap import plot_heatmap
    from integrated_hessians.simulation.plots.training_metrics import plot_training_metrics
    from integrated_hessians.simulation.plots.interaction import plot_interaction_subsetted

    from integrated_hessians import get_hessian, get_integrated_hessians
    from integrated_hessians.simulation.test_model import (
        get_test_data,
        plot_gif_hessians_from_baseline_to_real,
        get_prediction,
        get_attributions,
        interpolate_onehot,
        subset_onehot_hessian,
    )

    return (
        CNNMLP,
        NUCLEOTIDE_ORDER,
        SimulatedSequence,
        get_attributions,
        get_hessian,
        get_integrated_hessians,
        get_prediction,
        get_test_data,
        interpolate_onehot,
        plot_heatmap,
        plot_interaction_subsetted,
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
def _(Path, config_paths, json, simulation_dropdown):
    chosen_config_path = config_paths[simulation_dropdown.value]
    with open(chosen_config_path,"r") as f:
        config = json.load(f)
    TEST_DATA = Path(config["TEST_DATA"])
    OUT_BEST_MODEL = config["OUT_BEST_MODEL"]
    SEQLEN = config["SEQLEN"]
    TRAIN_DATA = config["TRAIN_DATA"]
    MODEL_WIDTH_MULTIPLIER = config["MODEL_WIDTH_MULTIPLIER"]
    return MODEL_WIDTH_MULTIPLIER, OUT_BEST_MODEL, SEQLEN, TEST_DATA


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Metrics
    """)
    return


@app.cell
def _(evaluation_paths, json, plot_training_metrics, simulation_dropdown):
    # Updated file path per your request
    with open(evaluation_paths[simulation_dropdown.value]) as ff:
        data = json.load(ff)
    plot_training_metrics(
        "Training metrics",
        data["train_epoch_losses"],
        data["train_step_losses"],
        data["val_epoch_losses"],
        data["val_step_losses"],
        data["val_r2_per_epoch"],
        data["val_mae_per_epoch"],
    )
    return


@app.cell
def _(SEQLEN, TEST_DATA, get_test_data):
    test_data = get_test_data(TEST_DATA, SEQLEN)
    return (test_data,)


@app.cell
def _(all_phens, all_preds, plt, r2_score):
    plt.scatter(all_phens, all_preds, alpha=0.01)
    r2 = r2_score(all_phens, all_preds)
    plt.title(f"phens vs preds. R2: {r2}")
    return


@app.cell
def _(model, test_data, torch):
    onehot_all = torch.tensor([x.one_hot for x in test_data], dtype=torch.float)
    all_phens = [x.phenotype for x in test_data]
    with torch.no_grad():
        all_preds = model(onehot_all)
    return all_phens, all_preds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretability
    """)
    return


@app.cell
def _(mo, test_data):
    set_of_names = sorted(list(set([x.motif_names[0] for x in test_data])))
    mo.vstack(["Motif names:", set_of_names])
    return (set_of_names,)


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
def _(mo, set_of_names):
    motif1_choose = mo.ui.dropdown(
        options=set_of_names + ["Any"], value="Any", label="Motif1:"
    )
    motif2_choose = mo.ui.dropdown(
        options=set_of_names + ["Any"], value="Any", label="Motif2:"
    )
    mo.vstack(["Subset sequences",motif1_choose, motif2_choose])
    return motif1_choose, motif2_choose


@app.cell
def _(mo):
    SELECTEd_ROW = mo.ui.number(0, 200, 1, label="Choose Row")
    SELECTEd_ROW
    return (SELECTEd_ROW,)


@app.cell
def _(
    CNNMLP,
    MODEL_WIDTH_MULTIPLIER,
    OUT_BEST_MODEL,
    SELECTEd_ROW,
    SEQLEN,
    SimulatedSequence,
    subsetted_data,
    torch,
):
    test_row: SimulatedSequence = subsetted_data[SELECTEd_ROW.value]
    model = CNNMLP(sequence_length=SEQLEN, width_multiplier=MODEL_WIDTH_MULTIPLIER)
    model.load_state_dict(torch.load(OUT_BEST_MODEL))
    model.eval()
    None
    return model, test_row


@app.cell
def _(
    Tensor,
    get_attributions,
    get_prediction,
    jx,
    model,
    sampling_options,
    test_row: "SimulatedSequence",
    torch,
):
    one_hot: jx.Float[Tensor, "1 alphabet_length sequence_length"] = torch.from_numpy(test_row.one_hot).unsqueeze(0).type(torch.float)
    mask1 = torch.tensor([int(x) for x in test_row.motif_mask_1])
    mask2 = torch.tensor([int(x) for x in test_row.motif_mask_2])

    match sampling_options.value:
        case "None":
            pass
        case "Keep motif1, shuffle the rest":
            shuffle_indices = torch.where(~mask1.bool())[0] # only shuffle outside mask1
            permuted_indices = shuffle_indices[torch.randperm(shuffle_indices.size(0))]
            shuffled_one_hot = one_hot.clone()
            shuffled_one_hot[:, shuffle_indices, :] = one_hot[:, permuted_indices, :]
            one_hot = shuffled_one_hot

            mask2 = torch.zeros_like(mask2)
        case "Keep motif2, shuffle the rest":
            shuffle_indices = torch.where(~mask2.bool())[0] # only shuffle outside mask1
            permuted_indices = shuffle_indices[torch.randperm(shuffle_indices.size(0))]
            shuffled_one_hot = one_hot.clone()
            shuffled_one_hot[:, shuffle_indices, :] = one_hot[:, permuted_indices, :]
            one_hot = shuffled_one_hot

            mask1 = torch.zeros_like(mask1)
        case "Shuffle Everything":
            #shuffling, this is only suitable for batch size 1 tensors as otherwise all samples will be shuffled the same
            # Generate random indices for the target dimension (dim 1)
            shuffling_indices = torch.randperm(one_hot.size(1))
            # Index into the tensor along dimension 1
            one_hot = one_hot[:, shuffling_indices, :]

            mask1 = torch.zeros_like(mask1)
            mask2 = torch.zeros_like(mask2)
        case _:
            raise Exception("Invalid option")


    row_prediction = get_prediction(model=model, batched_one_hot_input=one_hot)
    attributions, ig_delta = get_attributions(model=model, batched_one_hot_input=one_hot)
    return attributions, ig_delta, mask1, mask2, one_hot, row_prediction


@app.cell
def _():
    # attributions_permuted: jx.Float[
    #     NDArray[np.float32], "alphabet_length sequence_length"
    # ] = attributions.squeeze(0).numpy().transpose((1, 0))
    return


@app.cell
def _(
    model,
    one_hot: "jx.Float[Tensor, \"1 alphabet_length sequence_length\"]",
    plt,
    torch,
):
    from integrated_hessians import _get_interpolation_coefficients
    interpolation_coeffs = _get_interpolation_coefficients(approximation_steps=20,mode="default", verbose=True)[0]
    interpolation_coeffs = torch.tensor(interpolation_coeffs)
    interpolation_coeffs.shape
    
    onehot_baseline = torch.full_like(one_hot.squeeze(0),fill_value=.25)
    direction_baseline_to_input = (one_hot.squeeze(0) - onehot_baseline)
    path_from_baseline_to_input = (interpolation_coeffs.reshape(-1,1,1) * direction_baseline_to_input) + onehot_baseline

    with torch.no_grad():
        pred_path_from_baseline_to_input = model(path_from_baseline_to_input.type(torch.float))
    pred_path_from_baseline_to_input = pred_path_from_baseline_to_input.squeeze(-1)

    interpolation_preds_plot_fig, interpolation_preds_plot_ax = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
    interpolation_preds_plot_ax[0].scatter(interpolation_coeffs,pred_path_from_baseline_to_input,alpha=.1,s=50)
    interpolation_preds_plot_ax[0].set_title("From baseline to input")
    interpolation_preds_plot_ax[0].set_ylabel("prediction")
    interpolation_preds_plot_ax[0].set_xlabel("interpolation coeff")

    interpolation_preds_plot_ax[1].plot(path_from_baseline_to_input[:,0,:])
    interpolation_preds_plot_ax[1].set_title("Interpolation values for nucleotides")
    interpolation_preds_plot_ax[1].set_ylabel("Value of nucleotide")
    interpolation_preds_plot_ax[1].set_xlabel("Interpolation index")
    return


@app.cell
def _(
    get_hessian,
    jx,
    model,
    one_hot: "jx.Float[Tensor, \"1 alphabet_length sequence_length\"]",
    torch,
):
    calculated_hessian: jx.Float[
        torch.Tensor,
        "batch_size alphabet_length sequence_length batch_size alphabet_length sequence_length",
    ] = get_hessian(model=model, input=one_hot, target=0)
    # batch size is 1, so remove that dimension
    calculated_hessian: jx.Float[
        torch.Tensor,
        "alphabet_length sequence_length alphabet_length sequence_length",
    ] = calculated_hessian.squeeze(0).squeeze(2)
    return


@app.cell
def _(
    NUCLEOTIDE_ORDER,
    fig_genomic_line_height,
    figs_common_width,
    one_hot: "jx.Float[Tensor, \"1 alphabet_length sequence_length\"]",
    plot_heatmap,
    row_prediction,
    test_row: "SimulatedSequence",
):
    # - One hot encoded sequence heatmap
    plot_heatmap(
        one_hot[0].permute((1, 0)),
        cmap="Grays", 
        title=f"Phen: {test_row.phenotype} Pred: {row_prediction: .3}",
        fig_height=fig_genomic_line_height*4, 
        fig_width=figs_common_width, 
        row_labels=NUCLEOTIDE_ORDER,
        col_labels=list(test_row.nucleotides),
        add_colorbar=False
    )
    return


@app.cell
def _(
    fig_genomic_line_height,
    figs_common_width,
    mask1,
    mask2,
    mo,
    plot_heatmap,
    test_row: "SimulatedSequence",
):
    # - Annotate motif 1 and motif 2 location in heatmap, label their names/roles
    motif1 = plot_heatmap(
        mask1.unsqueeze(0),
        title=test_row.motif_names[0], 
        row_labels=[" "], # for well alignment with onehot plot
        cmap="Grays",
        fig_height=fig_genomic_line_height, 
        fig_width=figs_common_width
        ,add_colorbar=False)
    motif2 = plot_heatmap(
        mask2.unsqueeze(0),
        title=test_row.motif_names[1], 
        row_labels=[" "],
        cmap="Grays",
        fig_height=fig_genomic_line_height, 
        fig_width=figs_common_width
        ,add_colorbar=False)
    mo.vstack([motif1, motif2])
    return


@app.cell
def _(mo):
    sampling_options = mo.ui.radio(options=["None","Keep motif1, shuffle the rest","Keep motif2, shuffle the rest","Shuffle Everything"], label="Sampling Options", value="None")
    sampling_options
    return (sampling_options,)


@app.cell
def _(
    NUCLEOTIDE_ORDER,
    attributions,
    figs_common_width,
    ig_delta,
    plot_heatmap,
    test_row: "SimulatedSequence",
):
    # - One hot encoded Integrated Gradients
    plot_heatmap(
        attributions.squeeze(0).numpy().transpose((1, 0)),
        cmap="bwr", 
        title=f"Integrated Gradients (Multiplied input: true), delta: {float(ig_delta): .3f}",
        fig_height=1, 
        fig_width=figs_common_width, 
        row_labels=NUCLEOTIDE_ORDER,
        col_labels=list(test_row.nucleotides)
    )
    None # Uncomment this to show integrated gradients in one hot form
    return


@app.cell
def _(
    attributions,
    fig_genomic_line_height,
    figs_common_width,
    ig_delta,
    plot_heatmap,
    test_row: "SimulatedSequence",
):
    # - Dimensional reduced Integrated Gradients
    plot_heatmap(
        attributions.squeeze(0).sum(dim=1)[None,:].numpy(),
        cmap="bwr", 
        title=f"Integrated Gradients first dim summed (Real Positions), delta: {float(ig_delta): .3f}",
        fig_height=fig_genomic_line_height, 
        fig_width=figs_common_width, 
        row_labels=[" "],
        col_labels=list(test_row.nucleotides)
    )
    return


@app.cell
def _(attributions, mask1, mask2, test_row: "SimulatedSequence"):
    (
        f"First motif with name: {test_row.motif_names[0]} has attr sum: {(attributions[0] * mask1[:,None]).sum(): .3f}",
        f"Second motif with name: {test_row.motif_names[1]} has attr sum: {(attributions[0] * mask2[:,None]).sum(): .3f}",
    )
    return


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
    one_hot: "jx.Float[Tensor, \"1 alphabet_length sequence_length\"]",
    plot_interaction_subsetted,
    show_hessian,
    subset_onehot_hessian,
    torch,
):
    interpolation = interpolate_onehot(
        one_hot, baseline_to_input_alpha.value
    )

    interpolation_hessian: jx.Float[
        torch.Tensor,
        "alphabet_length sequence_length alphabet_length sequence_length",
    ] = get_hessian(
        model=model, input=interpolation, target=0
    )

    interpolation_hessian = interpolation_hessian.squeeze(0).squeeze(2)
    interpolation_hessian.shape

    pred_interpolation = get_prediction(model, interpolation)

    hessian_interaction_plot = plot_interaction_subsetted(
        hessian_onehot_subsetted=subset_onehot_hessian(
            calculated_hessian=interpolation_hessian,
            one_hot_mask=one_hot[0],
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
def _():
    # sampling_options = mo.ui.radio(options=["None","Keep motif1, shuffle the rest","Keep motif2, shuffle the rest","Shuffle Everything"], label="Sampling Options", value="None")
    # sampling_options
    return


@app.cell
def _(
    get_integrated_hessians,
    model,
    one_hot: "jx.Float[Tensor, \"1 alphabet_length sequence_length\"]",
    plot_interaction_subsetted,
    plt,
    sampling_steps,
    show_integrated_hessian,
    subset_onehot_hessian,
    torch,
):
    baseline_fill = 0.25
    if show_integrated_hessian.value:
        integ_hess_result, ih_delta = get_integrated_hessians(
            model=model,
            inputs=one_hot,
            baselines=torch.full_like(one_hot, baseline_fill),
            target=0,
            approximation_steps=sampling_steps.value,
            optimize_for_duplicate_interpolation_values=True,
        )

        integ_hess_plots_fig, integ_hess_plots_ax = plt.subplots(
            nrows=1, ncols=2, figsize=(10, 4)
        )

        ih_interaction_plot_subsetted = plot_interaction_subsetted(
            hessian_onehot_subsetted=subset_onehot_hessian(
                calculated_hessian=integ_hess_result.squeeze(0),
                one_hot_mask=one_hot[0],
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
    return baseline_fill, integ_hess_result


@app.cell
def _():
    return


@app.cell
def _(integ_hess_result, show_integrated_hessian):
    from integrated_hessians.simulation.plots.interaction import plot_genomic_interaction
    plot_genomic_interaction(
        integ_hess_result.squeeze(0).permute(0,2,1,3),
        fig_height=50,fig_width=50
    ) if show_integrated_hessian.value else None
    return


@app.cell
def _(
    NUCLEOTIDE_ORDER,
    figs_common_width,
    ih,
    plot_heatmap,
    show_integrated_hessian,
    test_row: "SimulatedSequence",
):
    #trying to get diagonals
    if show_integrated_hessian.value:
        tmp = ih.clone()
        print(tmp.shape)
        # tmp[:,3,:,0] = 1 # testing
        tmp = tmp.diagonal(dim1=0,dim2=2)
        row_labels = []
        for n1 in NUCLEOTIDE_ORDER:
            for n2 in NUCLEOTIDE_ORDER:
                row_labels.append(n1+n2)
    plot_heatmap(
        tmp.reshape(16,100),
        fig_width=figs_common_width, 
        title="Diagonals of IH",
        row_labels=row_labels,
        col_labels=list(test_row.nucleotides)
    ) if show_integrated_hessian.value else None
    return


@app.cell
def _(
    SEQLEN,
    integ_hess_result,
    mask1,
    mask2,
    mo,
    one_hot: "jx.Float[Tensor, \"1 alphabet_length sequence_length\"]",
    show_integrated_hessian,
    subset_onehot_hessian,
):
    if show_integrated_hessian.value:
        ih = integ_hess_result[0]
        ihrealnp = (
            subset_onehot_hessian(
                calculated_hessian=integ_hess_result.squeeze(0),
                one_hot_mask=one_hot[0],
            )
        )

        # ihrealnp_masked_sum_pair1 = (
        #     ihrealnp * mask1.reshape(SEQLEN, 1) * mask2.reshape(1, SEQLEN)
        # ).sum()
        # ihrealnp_masked_sum_pair2 = (
        #     ihrealnp * mask2.reshape(SEQLEN, 1) * mask1.reshape(1, SEQLEN)
        # ).sum()

        ih_masked_sum_pair1 = (
            ih * mask1.reshape(SEQLEN, 1, 1, 1) * mask2.reshape(1, 1, SEQLEN, 1)
        ).sum()
        ih_masked_sum_pair2 = (
            ih * mask1.reshape(1, 1, SEQLEN, 1) * mask2.reshape(SEQLEN, 1, 1, 1)
        ).sum()

        # ihrealnp_masked_selfinteractmotif1 = (
        #     ihrealnp * mask1.reshape(SEQLEN, 1) * mask1.reshape(1, SEQLEN)
        # ).sum()

        # ihrealnp_masked_selfinteractmotif2 = (
        #     ihrealnp * mask2.reshape(SEQLEN, 1) * mask2.reshape(1, SEQLEN)
        # ).sum()

        ih_masked_selfinteractmotif1 = (
            ih * mask1.reshape(1, 1, SEQLEN, 1) * mask1.reshape(SEQLEN, 1, 1, 1)
        ).sum()

        ih_masked_selfinteractmotif2 = (
            ih * mask2.reshape(1, 1, SEQLEN, 1) * mask2.reshape(SEQLEN, 1, 1, 1)
        ).sum()

        # allrealsum = (
        #     ihrealnp_masked_sum_pair1
        #     + ihrealnp_masked_sum_pair2
        #     + ihrealnp_masked_selfinteractmotif1
        #     + ihrealnp_masked_selfinteractmotif2
        # )

        all_sum = (
            ih_masked_sum_pair1
            + ih_masked_sum_pair2
            + ih_masked_selfinteractmotif1
            + ih_masked_selfinteractmotif2
        )

    mo.vstack([
        "For integrated hessian:",
        # f"{ihrealnp_masked_sum_pair1 = :.3f}",
        # f"{ihrealnp_masked_sum_pair2 = :.3f}",
        # f"{ihrealnp_masked_sum_pair1 + ihrealnp_masked_sum_pair2 = :.3f}",
        # f"{ihrealnp_masked_selfinteractmotif1 = :.3f}",
        # f"{ihrealnp_masked_selfinteractmotif2 = :.3f}",
        # f"{allrealsum = :.3f}",
        # f"{ihrealnp.sum() = :.3f}",
        f"Overall sum of integrated hessian is {integ_hess_result.sum() :.3f}",
        # f"{ih_masked_sum_pair1 = :.3f}",
        # f"{ih_masked_sum_pair2 = :.3f}",
        # f"{ih_masked_sum_pair1 + ih_masked_sum_pair2 = :.3f}",
        f"Sum of first motif pair region {ih_masked_sum_pair1:.3f}",
        f"Sum of second motif pair region {ih_masked_sum_pair2:.3f}",
        f"Sum of both Motif Pair regions in {ih_masked_sum_pair1 + ih_masked_sum_pair2:.3f}",
        f"IH sum for motif1 only (self interaction) {ih_masked_selfinteractmotif1 :.3f}",
        f"IH sum for motif2 only (self interaction)  {ih_masked_selfinteractmotif2 :.3f}",
        f"Sum of self interacitons and pair interactions: {all_sum:.3f}",
    ]) if show_integrated_hessian.value else None
    return (
        all_sum,
        ih,
        ih_masked_selfinteractmotif1,
        ih_masked_selfinteractmotif2,
        ih_masked_sum_pair1,
        ih_masked_sum_pair2,
    )


@app.cell
def _(
    all_sum,
    ih_masked_selfinteractmotif1,
    ih_masked_selfinteractmotif2,
    ih_masked_sum_pair1,
    ih_masked_sum_pair2,
    integ_hess_result,
    np,
    plt,
    show_integrated_hessian,
):
    if show_integrated_hessian.value:
        attr_plot_data = np.array(
            [
                [
                    ih_masked_selfinteractmotif1,
                    ih_masked_sum_pair1
                ],[
                    ih_masked_sum_pair2,
                    ih_masked_selfinteractmotif2
                ]]
        )
        attr_plot_fig, attr_plot_ax = plt.subplots(figsize=(6, 6))
        attr_plot_ax.imshow(attr_plot_data, cmap="bwr", vmin=-1, vmax=1)
        attr_plot_ax.text(0,0,f"self interact 1: {ih_masked_selfinteractmotif1 :.3f}",ha="center",va="center",fontsize=14,)
        attr_plot_ax.text(1,0,f"pair1: {ih_masked_sum_pair1: .3f}",ha="center",va="center",fontsize=14,)
        attr_plot_ax.text(0,1,f"pair2: {ih_masked_sum_pair2: .3f}",ha="center",va="center",fontsize=14,)
        attr_plot_ax.text(1,1,f"self interact 2: {ih_masked_selfinteractmotif2 :.3f}",ha="center",va="center",fontsize=14,)
        attr_plot_ax.set_xticks([0, 1])
        attr_plot_ax.set_xticklabels(["motif1", "motif2"])
        attr_plot_ax.set_yticks([0, 1])
        attr_plot_ax.set_yticklabels(["motif1", "motif2"])
        attr_plot_ax.set_title(f"Sums of Regions in IH \nOverall IH Sum: {integ_hess_result.sum() :.3f} \nSum of these four regions: {all_sum:.3f}")
        attr_plot_ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
        attr_plot_ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        attr_plot_ax.grid(which="minor", color="black", linewidth=1)
        attr_plot_ax.tick_params(which="minor", length=0)
    attr_plot_ax if show_integrated_hessian.value else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Janizek comparison
    """)
    return


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
    baseline_fill,
    model,
    one_hot: "jx.Float[Tensor, \"1 alphabet_length sequence_length\"]",
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
        exp_baseline = torch.full_like(exp_input, baseline_fill)
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
            - model(torch.full_like(one_hot_batched, baseline_fill))
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
