import marimo

__generated_with = "0.23.1"
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
def _(json, pl, rootpath):
    with open(rootpath/"implementation_performance_comparison.json") as f2:
        perf_comparison = json.load(f2)
    perf_comparison_df = pl.concat([pl.from_dicts(perf_comparison[impl_name]).with_columns(implementation=pl.lit(impl_name)) for impl_name in perf_comparison])
    perf_comparison_df
    return (perf_comparison_df,)


@app.cell
def _(perf_comparison_df, plt, rootpath):
    plt.scatter(perf_comparison_df["implementation"],perf_comparison_df["delta"])
    plt.title("deltas")
    plt.savefig(rootpath / "implementations_vs_deltas.svg")
    plt.gcf()
    return


@app.cell
def _(perf_comparison_df, plt, rootpath):
    plt.scatter(perf_comparison_df["implementation"],perf_comparison_df["comptime_seconds"])
    plt.yscale('log')
    plt.ylabel("log")
    plt.title("comptime_seconds")
    plt.savefig(rootpath / "implementations_vs_comptime_seconds.svg")
    plt.gcf()
    return


@app.cell
def _(perf_comparison_df, plt):
    plt.scatter(perf_comparison_df["implementation"],perf_comparison_df["function_calls"])
    plt.title("Function calls")
    # plt.savefig(rootpath / "implementations_vs_function_calls.svg")
    plt.gcf()
    return


@app.cell
def _(perf_comparison_df, plt):
    plt.scatter(perf_comparison_df["implementation"],perf_comparison_df["delta"]/(1/perf_comparison_df["function_calls"]))
    plt.title("Try to see if ratios lead to something meaningful")
    return


if __name__ == "__main__":
    app.run()
