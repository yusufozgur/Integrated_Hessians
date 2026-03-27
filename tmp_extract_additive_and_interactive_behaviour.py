import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from integrated_hessians.simulation.custom_additive_and_interactive_effects.test_model import get_test_data
    from integrated_hessians.simulation.custom_additive_and_interactive_effects.config import TEST_DATA
    import torch

    return TEST_DATA, get_test_data, torch


@app.cell
def _(TEST_DATA, get_test_data, torch):
    test_data = get_test_data(TEST_DATA)
    torch.tensor([int(x) for x in test_data[0].motif_mask_1], dtype=torch.bool)
    return


if __name__ == "__main__":
    app.run()
