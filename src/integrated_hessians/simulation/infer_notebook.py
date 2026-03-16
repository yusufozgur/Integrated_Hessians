import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot
    """)
    return


@app.cell
def _(mo):
    row = mo.ui.number(0,1000,1, label="Choose Row")
    row
    return (row,)


@app.cell
def _(mo):
    baseline_to_input_alpha = mo.ui.slider(0,1,.1, show_value=True, label="Baseline to input alpha")
    baseline_to_input_alpha
    return (baseline_to_input_alpha,)


@app.cell
def _(
    attributions_permuted,
    h,
    interpolation_pred,
    onehot_permuted,
    plot_binary_string,
    plot_heatmap,
    plot_onehot,
    plt,
    pred,
    real_attributions,
    seq,
):
    fig, axes = plt.subplots(
        nrows=6,
        figsize=(10, 10),
        sharex=True,                          # guarantees column alignment
        gridspec_kw={'height_ratios': [4, 1, 1, 4, 1, 20]} # optional: scale row heights by number of rows
    )

    plot_onehot(seq.nucleotides, onehot_permuted, axes[0], title=f"Phen: {seq.phenotype} Pred: {pred: .2}")
    plot_binary_string(seq.motif_mask_1, axes[1], title=seq.motif_types[0])
    plot_binary_string(seq.motif_mask_2, axes[2], title=seq.motif_types[1])
    plot_onehot(seq.nucleotides, attributions_permuted, axes[3], title="Integrated Gradients (Multiplied input true)")
    plot_heatmap(real_attributions, row_labels=["Real base"], col_labels=list(seq.nucleotides), ax=axes[4], cmap='bwr', title="Real Attributions")

    import matplotlib.colors as mcolors
    norm = mcolors.TwoSlopeNorm(vmin=h.min(), vcenter=0, vmax=h.max())
    im = axes[5].imshow(h, aspect='auto', cmap='bwr', norm=norm)
    axes[5].set_title(f"Interpolation phenotype: {interpolation_pred: .2f}")
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[5])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)

    axes[0].set_xlabel("")  # remove redundant x-label from top plot
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(hess, plt):
    plt.imshow(hess.reshape(200,200),cmap='bwr')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import matplotlib.pyplot as plt
    import torch
    from captum.attr import IntegratedGradients
    import numpy as np

    return IntegratedGradients, Path, mo, np, plt, torch


@app.cell
def _():
    from integrated_hessians.simulation.train_model import MotifInteractionsDataset
    from integrated_hessians.simulation.simple_simulation.model import CNNDense
    from integrated_hessians.simulation.plot import plot_onehot, plot_binary_string, plot_heatmap
    from integrated_hessians.simulation import MotifType

    return (
        CNNDense,
        MotifInteractionsDataset,
        MotifType,
        plot_binary_string,
        plot_heatmap,
        plot_onehot,
    )


@app.cell
def _(MotifInteractionsDataset, Path):
    seqs = MotifInteractionsDataset(Path("data/simple_simulation/1k_test.json")).data
    return (seqs,)


@app.cell
def _(row, seqs):
    seq = seqs[row.value]
    return (seq,)


@app.cell
def _(CNNDense, torch):
    model = CNNDense()
    model.load_state_dict(torch.load("data/simple_simulation/model_best.pth"))
    model.eval()
    return (model,)


@app.cell
def _(model, seq, torch):
    # Calculate Model Prediction
    with torch.no_grad():
        pred = model(torch.tensor(seq.one_hot).unsqueeze(0).type(torch.float))
    pred = float(pred[0,0])
    return (pred,)


@app.cell
def _(IntegratedGradients, model, seq, torch):
    #Calculate Attributions via Integrated Gradients
    input = torch.tensor(seq.one_hot).unsqueeze(0).type(torch.float)
    baseline = torch.full_like(input, 0)
    ig = IntegratedGradients(model, multiply_by_inputs=True)
    attributions, delta = ig.attribute(input, baseline, return_convergence_delta=True)
    f"Delta: {float(delta): .2}"
    return attributions, input


@app.cell
def _(attributions, np, seq):
    # Get attributions for existent positions (One hot -> Real)
    onehot_permuted = np.permute_dims(seq.one_hot, [1,0])
    attributions_permuted = np.permute_dims(attributions.squeeze(0).numpy(),[1,0])
    real_attributions = np.sum(onehot_permuted * attributions_permuted, axis=0).reshape(1, -1)
    return attributions_permuted, onehot_permuted, real_attributions


@app.cell
def _(baseline_to_input_interpolation, model):
    # Calculate Hessian
    from integrated_hessians.hessian import hessian
    hess = hessian(model, baseline_to_input_interpolation, target=0)
    hess.shape
    return (hess,)


@app.cell
def _(baseline_to_input_alpha, input, interpolate_onehot, model, torch):
    baseline_to_input_interpolation = interpolate_onehot(input, baseline_to_input_alpha.value)
    with torch.no_grad():
        interpolation_pred = model(baseline_to_input_interpolation)
    interpolation_pred = float(interpolation_pred[0,0])
    return baseline_to_input_interpolation, interpolation_pred


@app.cell
def _(baseline_to_input_interpolation):
    baseline_to_input_interpolation
    return


@app.cell
def _(torch):
    def interpolate_onehot(onehot_tensor, alpha):
        """
        Interpolate between uniform (0.25) and one-hot encoded tensor.

        Args:
            onehot_tensor: shape (50, 4) one-hot encoded tensor
            alpha: float in [0, 1], where 0 = uniform, 1 = one-hot

        Returns:
            Interpolated tensor of shape (50, 4)
        """
        uniform = torch.full_like(onehot_tensor, 0.25)
        return (1 - alpha) * uniform + alpha * onehot_tensor

    return (interpolate_onehot,)


@app.cell
def _(hess, seq, torch):
    idx = torch.tensor(seq.one_hot).bool()          # [50, 4]
    flat_idx = idx.nonzero(as_tuple=False)[:, 1]    # [50] — which of the 4 cols is hot, per row

    h = hess[0]                    # [50, 4, 1, 50, 4]
    h = h[idx]                     # [50, 1, 50, 4]
    h = h[:, 0, :, :]              # [50, 50, 4]

    # For each of the 50 rows in the last dim, pick the hot column
    h = h[torch.arange(50), :, :]                   # still [50, 50, 4]
    h = h[:, torch.arange(50), flat_idx]            # [50, 50] ✓
    return (h,)


@app.cell
def _(h):
    h.shape
    return


@app.cell
def _(MotifType, mo):
    # Search Dataset
    search1 = mo.ui.dropdown(options=list(MotifType))
    search2 = mo.ui.dropdown(options=list(MotifType))
    "Search in dataset", search1, search2
    return search1, search2


@app.cell
def _(search1, search2, seqs):
    for i,x in enumerate(seqs):
        if x.motif_types[0] == search1.value:
            if x.motif_types[1] == search2.value:
                print(f"Found specified datapoint at {i}")
                break
    return


if __name__ == "__main__":
    app.run()
