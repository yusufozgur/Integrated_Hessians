import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import polars as pl
    import numpy as np
    from sklearn.linear_model import Ridge
    from integrated_hessians.simulation.randomized_additive_and_interactive_effects.config import OUT_EXTRACTED_self_interactions_and_pair_interactions_sums, OUT_ADDITIVE_DEFINED_EFFECTS, OUT_INTERACTIVE_DEFINED_EFFECTS
    import matplotlib.pyplot as plt
    import seaborn as sns

    return (
        OUT_ADDITIVE_DEFINED_EFFECTS,
        OUT_EXTRACTED_self_interactions_and_pair_interactions_sums,
        OUT_INTERACTIVE_DEFINED_EFFECTS,
        json,
        mo,
        np,
        pl,
        plt,
        sns,
    )


@app.cell
def _():
    sorted(["Motif7","Motif5"])
    return


@app.cell
def _(
    OUT_ADDITIVE_DEFINED_EFFECTS,
    OUT_EXTRACTED_self_interactions_and_pair_interactions_sums,
    OUT_INTERACTIVE_DEFINED_EFFECTS,
    json,
):
    with open(OUT_EXTRACTED_self_interactions_and_pair_interactions_sums) as f:
        data = json.load(f)
    with open(OUT_ADDITIVE_DEFINED_EFFECTS) as f:
        defined_additive = json.load(f)
    with open(OUT_INTERACTIVE_DEFINED_EFFECTS) as f:
        defined_interactive = json.load(f)
    # add order insensitive pair names
    data = [x|{"pairing":"_".join(sorted([x["name1"],x["name2"]]))} for x in data]
    data[:1]
    return data, defined_additive, defined_interactive


@app.cell
def _(data, pl):
    df = pl.DataFrame(data)
    # df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extracting Additive Effects

    We take mean of self interaction sums of the integrated hessians for motifs. For purely additive motifs, this should be equal to integrated gradients motif sums, but for motifs with interactive components, integrated hessians self interaction is more accurate.
    """)
    return


@app.cell
def _(df, pl):
    motif1_and_2_seperate_index_cols = ["sum_of_pairs", "prediction", "phenotype", "pairing"]
    df_motif1_and_2_seperated = pl.concat([
        df.select(motif1_and_2_seperate_index_cols + [pl.col("name1").alias("name"), pl.col("sum_self_interaction_1").alias("self_interaction")]),
        df.select(motif1_and_2_seperate_index_cols + [pl.col("name2").alias("name"), pl.col("sum_self_interaction_2").alias("self_interaction")])
    ])
    # df_motif1_and_2_seperated
    return (df_motif1_and_2_seperated,)


@app.cell
def _(df_motif1_and_2_seperated, mo, pl):
    df_additive_effects = (
        df_motif1_and_2_seperated
            .group_by("name")
            .agg(
                pl.col("self_interaction").mean().alias("mean_self_interaction")
            )
            .sort("name")
    )
    additive_effects = dict(zip(df_additive_effects["name"],df_additive_effects["mean_self_interaction"]))
    mo.vstack(["additive effects",additive_effects])
    return (df_additive_effects,)


@app.cell
def _(defined_additive, pl, plt, sns):

    plt.figure(figsize=(8, 2))
    sns.heatmap(pl.from_dict(defined_additive).to_pandas(), annot=True, cmap='Blues', center=0, cbar_kws={'label': 'Self Interaction Score'})
    plt.title('Additive Effects: Definitions')
    plt.gca()
    return


@app.cell
def _(df_additive_effects, plt, sns):
    plt.figure(figsize=(8, 2))
    sns.heatmap(df_additive_effects.to_pandas().set_index('name').T, annot=True, cmap='Blues', center=0, cbar_kws={'label': 'Self Interaction Score'})
    plt.title('Additive Effects: Mean self interactions')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extracting Interaction Effects
    """)
    return


@app.cell
def _(df, pl):
    df_sum = df.with_columns(
            sum=pl.col("sum_of_pairs")+pl.col("sum_self_interaction_1")+pl.col("sum_self_interaction_2")
        ).with_columns(
            interactive_ratio=pl.col("sum_of_pairs")/pl.col("sum")
        )
    # df_sum
    return (df_sum,)


@app.cell
def _(df_sum, pl):
    names_and_sum = (
        df_sum
            .select("name1","name2","sum")
            .group_by(["name1","name2"])
            .agg(pl.col("sum").mean())
            .sort("name2")
            .pivot(on="name2",index="name1")
            .sort("name1",descending=False)
    )
    names_and_pairinteract = (
        df_sum
            .select("name1","name2","sum_of_pairs")
            .group_by(["name1","name2"])
            .agg(pl.col("sum_of_pairs").mean())
            .sort("name2")
            .pivot(on="name2",index="name1")
            .sort("name1",descending=False)
    )
    return (names_and_sum,)


@app.cell
def _(names_and_sum, plt, sns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        names_and_sum.drop('name1'), 
        annot=True,     # Shows the numeric values in cells
        cmap="Blues",  # Color palette
        fmt=".2f",      # Float precision
        linewidths=.5,
        xticklabels=names_and_sum.drop("name1").columns,
        yticklabels=names_and_sum["name1"]
    )
    plt.gca().xaxis.tick_top()
    plt.title("Mean AllSum (PairInteractionSum+SelfInteractionSum1,2)", pad=40)
    "This includes both interactive and additive effects",plt.gca()
    return


@app.cell
def _(df_additive_effects, df_sum, pl):
    df_sum_minus_additives = (
        df_sum
        .select("name1","name2","sum")
        .group_by(["name1","name2"])
        .agg(pl.col("sum").mean())
        .sort("name2")
        .join(
            df_additive_effects.rename(
                {"mean_self_interaction":"mean_self_interaction_name1"}
            ),left_on="name1",right_on="name")
        .join(
            df_additive_effects.rename(
                {"mean_self_interaction":"mean_self_interaction_name2"}),
            left_on="name2",right_on="name")
        .with_columns(
            sum_minus_additives=pl.col("sum") - pl.col("mean_self_interaction_name1") - pl.col("mean_self_interaction_name2")
        )
        .select("name1","name2","sum_minus_additives")
        .pivot(on="name2",index="name1")
        .sort("name1",descending=False)
    )
    return (df_sum_minus_additives,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This should only include interactive effects
    """)
    return


@app.cell
def _(df_sum_minus_additives, plt, sns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_sum_minus_additives.drop('name1'), 
        annot=True,     # Shows the numeric values in cells
        cmap="Blues",  # Color palette
        fmt=".2f",      # Float precision
        linewidths=.5,
        xticklabels=df_sum_minus_additives.drop("name1").columns,
        yticklabels=df_sum_minus_additives["name1"]
    )
    plt.gca().xaxis.tick_top()
    plt.title("Interactions: AllSum Minus MeanAdditives", pad=40)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For example, Motif6-1 was 0.70, when we subtracted additive component of 0.2 of motif6, it has become 0.5, real interaction value.
    """)
    return


@app.cell
def _(defined_interactive_matrix, defined_interactive_matrix_names, plt, sns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        defined_interactive_matrix, 
        annot=True,     # Shows the numeric values in cells
        cmap="Blues",  # Color palette
        fmt=".2f",      # Float precision
        linewidths=.5,
        xticklabels=defined_interactive_matrix_names,
        yticklabels=defined_interactive_matrix_names
    )
    plt.gca().xaxis.tick_top()
    plt.title("Interactions: Definitions", pad=40)
    plt.gca()
    return


@app.cell
def _(defined_interactive, np):
    defined_interactive_setkeys = {frozenset(k.split("_")):v for k,v in defined_interactive.items()}
    defined_interactive_matrix_names = sorted(list(set(("_".join(list(defined_interactive))).split("_"))))
    defined_interactive_matrix = np.zeros([len(defined_interactive_matrix_names),len(defined_interactive_matrix_names)])
    k = len(defined_interactive_matrix_names)
    for i,motif1 in enumerate(defined_interactive_matrix_names):
        for j,motif2 in enumerate(defined_interactive_matrix_names):
            if i == j:
                continue
            defined_interactive_matrix[i,j] = defined_interactive_setkeys[frozenset({motif1,motif2})]
    return defined_interactive_matrix, defined_interactive_matrix_names


if __name__ == "__main__":
    app.run()
