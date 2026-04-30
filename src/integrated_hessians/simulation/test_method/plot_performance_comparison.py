import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import polars as pl
    import matplotlib.pyplot as plt
    from pathlib import Path

    return Path, json, pl, plt


@app.cell
def _(Path):
    rootpath = Path("src/integrated_hessians/simulation/test_method/")
    return (rootpath,)


@app.cell
def _():
    fig_width = 12
    fig_height = 6
    return fig_width, fig_height


@app.cell
def _(json, pl, rootpath):
    with open(rootpath / "implementation_performance_comparison.json") as f2:
        perf_comparison = json.load(f2)
    perf_comparison_df = pl.concat(
        [
            pl.from_dicts(perf_comparison[impl_name]).with_columns(
                implementation=pl.lit(impl_name)
            )
            for impl_name in perf_comparison
        ]
    )
    perf_comparison_df
    return (perf_comparison_df,)


@app.cell
def _(perf_comparison_df, plt, rootpath, fig_width, fig_height):
    plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(perf_comparison_df["implementation"], perf_comparison_df["delta"])
    plt.ylabel("delta")
    plt.title("deltas")
    plt.savefig(rootpath / "implementations_vs_deltas.svg")
    plt.gcf()
    return


@app.cell
def _(perf_comparison_df, plt, rootpath, fig_width, fig_height):
    plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(
        perf_comparison_df["implementation"], perf_comparison_df["comptime_seconds"]
    )
    plt.ylabel("time")
    plt.title("comptime_seconds (linear y scale)")
    plt.savefig(rootpath / "implementations_vs_comptime_seconds_linear.svg")
    plt.gcf()
    return


@app.cell
def _(perf_comparison_df, plt, rootpath, fig_width, fig_height):
    plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(
        perf_comparison_df["implementation"], perf_comparison_df["comptime_seconds"]
    )
    plt.yscale("log")
    plt.ylabel("time (log)")
    plt.title("comptime_seconds (log y scale)")
    plt.savefig(rootpath / "implementations_vs_comptime_seconds_log.svg")
    plt.gcf()
    return


@app.cell
def _(perf_comparison_df, plt, fig_width, fig_height):
    plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(
        perf_comparison_df["implementation"], perf_comparison_df["function_calls"]
    )
    plt.title("Function calls")
    # plt.savefig(rootpath / "implementations_vs_function_calls.svg")
    plt.gcf()
    return


@app.cell
def _(perf_comparison_df, plt, fig_width, fig_height):
    plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(
        perf_comparison_df["implementation"],
        perf_comparison_df["delta"] / (1 / perf_comparison_df["function_calls"]),
    )
    plt.title("Try to see if ratios lead to something meaningful")
    return


if __name__ == "__main__":
    app.run()
