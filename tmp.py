import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return (Path,)


@app.cell
def _():
    from integrated_hessians.simulation.train_model import MotifInteractionsDataset

    return (MotifInteractionsDataset,)


@app.cell
def _(MotifInteractionsDataset, Path):
    dataset = MotifInteractionsDataset(input=Path("data/1k.json"))
    return (dataset,)


@app.cell
def _(dataset):
    next(iter(dataset))[0].shape
    return


if __name__ == "__main__":
    app.run()
