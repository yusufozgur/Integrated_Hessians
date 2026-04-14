import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import json
    # Updated file path per your request
    with open("data/simple_simulation/model_best_evaluation.json") as f:
        data = json.load(f)
    from integrated_hessians.simulation.plots.training_metrics import plot_training_metrics
    plot_training_metrics(
        data["train_epoch_losses"],
        data["train_step_losses"],
        data["val_epoch_losses"],
        data["val_step_losses"],
        data["val_r2_per_epoch"],
        data["val_mae_per_epoch"],
    )
    return


if __name__ == "__main__":
    app.run()
