import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Metrics
    """)
    return


@app.cell
def _():
    import json
    # Updated file path per your request
    with open("data/simple_simulation/model_best_evaluation.json") as f:
        data = json.load(f)
    from integrated_hessians.simulation.plots.training_metrics import plot_training_metrics
    plot_training_metrics(
        "title",
        data["train_epoch_losses"],
        data["train_step_losses"],
        data["val_epoch_losses"],
        data["val_step_losses"],
        data["val_r2_per_epoch"],
        data["val_mae_per_epoch"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Heatmaps
    """)
    return


@app.cell
def _(np):
    from integrated_hessians.simulation.plots.heatmap import plot_heatmap

    plot_heatmap(np.array([[0,1,2]]))
    return (plot_heatmap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Can be used for visualizing masks
    """)
    return


@app.cell
def _(np, plot_heatmap):
    plot_heatmap(np.array([[0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0]]),cmap="Grays", title="Mask from numpy", fig_height=.5, fig_width=10)
    return


@app.cell
def _(np, plot_heatmap):
    str_mask = "0000111110000"
    np_mask = np.array([list(map(int,list(str_mask)))])
    plot_heatmap(np_mask,cmap="Grays", title="Mask from str", fig_height=.5, fig_width=10)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Can be used to visualize one hot encoding
    """)
    return


@app.cell
def _(np, plot_heatmap):
    plot_heatmap(
        np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
        cmap="Grays", 
        title="One Hot", 
        fig_height=2, 
        fig_width=10, 
        row_labels=["A","C","G","T"],
        col_labels=[1,2,3,4]
    )
    return


if __name__ == "__main__":
    app.run()
