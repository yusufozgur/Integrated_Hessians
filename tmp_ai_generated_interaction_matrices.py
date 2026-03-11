import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    def plot_pcolormesh(Z, labels=None, cmap='viridis', title='pcolormesh Plot',
                       colorbar_label='Value', vmin=None, vmax=None,
                       figsize=(8, 6), triangle='lower'):
        """
        Plot a 2D numpy array as a 45-degree rotated triangle heatmap, pointed down.

        Parameters:
            Z        : 2D square numpy array to plot
            labels   : list of tick labels (optional)
            cmap     : colormap string (default 'viridis')
            title    : plot title
            colorbar_label : label for the colorbar
            vmin, vmax     : color scale limits (optional)
            figsize  : figure size tuple
            triangle : 'lower' (default) or 'upper'
        """
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
            raise ValueError(f"Z must be a 2D square array, got shape {Z.shape}")

        n = Z.shape[0]
        norm = plt.Normalize(vmin=vmin if vmin is not None else Z.min(),
                             vmax=vmax if vmax is not None else Z.max())
        cmap_obj = plt.get_cmap(cmap)

        patches = []
        values = []

        for i in range(n):
            for j in range(n):
                if triangle == 'lower' and j > i:
                    continue
                if triangle == 'upper' and j < i:
                    continue

                # Rotate 90°: swap cx/cy axes so triangle points down
                cx = (i + j)       # horizontal: sum → spreads left-right
                cy = (j - i)      # vertical:   diff → tip points down

                diamond = Polygon([
                    [cx,     cy + 1],
                    [cx + 1, cy    ],
                    [cx,     cy - 1],
                    [cx - 1, cy    ],
                ], closed=True)

                patches.append(diamond)
                values.append(Z[i, j])

        fig, ax = plt.subplots(figsize=figsize)

        col = PatchCollection(patches, cmap=cmap_obj, norm=norm, edgecolors='none')
        col.set_array(np.array(values))
        ax.add_collection(col)

        # Labels along the top
        if labels is not None:
            for k, label in enumerate(labels):
                ax.text(k * 2, 1.2, label, ha='center', va='bottom',
            fontsize=10, rotation=0)

        fig.colorbar(col, ax=ax, label=colorbar_label, fraction=0.03, pad=0.04)

        ax.set_xlim(-1.5, (n - 1) * 2 + 1.5)
        ax.set_ylim(-(n - 1) - 1.5, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, pad=20)

        plt.tight_layout()
        return ax


    # --- Example ---
    n = 5
    np.random.seed(42)
    Z = np.random.rand(n, n)
    Z = (Z + Z.T) / 2

    labels = [f'V{i}' for i in range(n)]
    plot_pcolormesh(Z, labels=labels, title='Rotated Interaction Matrix',
                    colorbar_label='Strength', triangle='lower')
    return PatchCollection, Polygon, np, plt


@app.cell
def _(PatchCollection, Polygon, np, plt):

    from matplotlib.colors import Normalize
    NUCLEOTIDES = ['A', 'C', 'G', 'T']

    def plot_genomic_interaction_heatmap(
        one_hot,
        Z=None,
        cmap='RdBu_r',
        title='Genomic Interaction Matrix',
        colorbar_label='Interaction Strength',
        vmin=None,
        vmax=None,
        figsize=(12, 9),
        triangle='lower',
        outer_gap=0.05,
        inner_gap=0.01,
    ):
        """
        Plot a 45-degree rotated triangle heatmap for a genomic sequence.

        Each outer diamond corresponds to a (position_i, position_j) pair.
        Inside each outer diamond is a 4×4 grid of inner diamonds, one per
        nucleotide combination (A,C,G,T × A,C,G,T).

        The color of each inner diamond is determined by Z[i, j, a, b],
        the interaction strength between nucleotide a at position i and
        nucleotide b at position j.

        Parameters
        ----------
        one_hot : np.ndarray, shape (L, 4)
            One-hot encoded sequence. Used to highlight the observed nucleotides
            (the 4 inner squares on the diagonal of the observed pair are outlined).
        Z : np.ndarray, shape (L, L, 4, 4), optional
            Interaction values. If None, a random symmetric example is generated.
        cmap : str
            Matplotlib colormap.
        title : str
            Plot title.
        colorbar_label : str
            Label for the colorbar.
        vmin, vmax : float, optional
            Color scale limits.
        figsize : tuple
            Figure size.
        triangle : str
            'lower' or 'upper'.
        outer_gap : float
            Fractional gap between outer diamonds (0–0.5).
        inner_gap : float
            Fractional gap between inner sub-diamonds (0–0.5).
        """
        one_hot = np.asarray(one_hot, dtype=float)
        if one_hot.ndim != 2 or one_hot.shape[1] != 4:
            raise ValueError(f"one_hot must have shape (L, 4), got {one_hot.shape}")
        L = one_hot.shape[0]

        # Build or validate Z
        if Z is None:
            rng = np.random.default_rng(0)
            Z = rng.standard_normal((L, L, 4, 4))
            # Make symmetric: Z[i,j,a,b] = Z[j,i,b,a]
            for i in range(L):
                for j in range(L):
                    Z[i, j] = (Z[i, j] + Z[j, i].T) / 2
                    Z[j, i] = Z[i, j].T
        else:
            Z = np.asarray(Z, dtype=float)
            if Z.shape != (L, L, 4, 4):
                raise ValueError(f"Z must have shape ({L},{L},4,4), got {Z.shape}")

        norm = Normalize(
            vmin=vmin if vmin is not None else Z.min(),
            vmax=vmax if vmax is not None else Z.max(),
        )
        cmap_obj = plt.get_cmap(cmap)

        # Each outer diamond is centred at (cx, cy) in "outer" coords.
        # We subdivide it into a 4×4 arrangement of inner diamonds.
        # The outer diamond has half-width 1 along each rotated axis.
        # We shrink by outer_gap so diamonds don't touch.
        outer_scale = 1.0 - outer_gap          # half-size of outer diamond
        inner_half  = outer_scale / 4          # half-size of one inner cell
        inner_scale = inner_half * (1 - inner_gap)  # shrink for inner gap

        # Sub-cell centres relative to outer diamond centre.
        # We tile 4 cells along the two 45° axes.
        # Axis 1 (→ right/up):  unit vector (1, 1)/√2 * 2*inner_half each step
        # Axis 2 (→ right/down): unit vector (1,-1)/√2 * 2*inner_half each step
        # For rows a=0..3, cols b=0..3:
        #   offset = (a - 1.5) * step_along_axis1 + (b - 1.5) * step_along_axis2
        # In screen coords (cx horizontal, cy vertical):
        #   axis1 step: (+inner_half, +inner_half)  [→ top-right]
        #   axis2 step: (+inner_half, -inner_half)  [→ bottom-right]
        step = 2 * inner_half  # distance between sub-cell centres along each axis

        patches = []
        colors  = []
        highlight_patches = []

        # Identify observed nucleotides per position
        obs = np.argmax(one_hot, axis=1)  # shape (L,)

        for i in range(L):
            for j in range(L):
                if triangle == 'lower' and j > i:
                    continue
                if triangle == 'upper' and j < i:
                    continue

                # Outer diamond centre (same formula as original)
                cx = float(i + j)
                cy = float(j - i)

                for a in range(4):        # nucleotide at position i  (rows)
                    for b in range(4):    # nucleotide at position j  (cols)
                        # Sub-cell centre relative to outer centre
                        # We want a to go "left to right" (increasing cx direction)
                        # and b to go "top to bottom" (decreasing cy direction)
                        da = a - 1.5
                        db = b - 1.5
                        # Shift along the two 45° axes
                        dx = (da + db) * inner_half
                        dy = (db - da) * inner_half

                        scx = cx + dx
                        scy = cy + dy

                        diamond = Polygon(
                            [
                                [scx,              scy + inner_scale],
                                [scx + inner_scale, scy             ],
                                [scx,              scy - inner_scale],
                                [scx - inner_scale, scy             ],
                            ],
                            closed=True,
                        )
                        patches.append(diamond)
                        colors.append(Z[i, j, a, b])

                        # Highlight the observed nucleotide pair
                        if a == obs[i] and b == obs[j]:
                            hl = Polygon(
                                [
                                    [scx,              scy + inner_scale],
                                    [scx + inner_scale, scy             ],
                                    [scx,              scy - inner_scale],
                                    [scx - inner_scale, scy             ],
                                ],
                                closed=True,
                            )
                            highlight_patches.append(hl)

        fig, ax = plt.subplots(figsize=figsize)

        # Main colored patches
        col = PatchCollection(patches, cmap=cmap_obj, norm=norm,
                              edgecolors='none', zorder=1)
        col.set_array(np.array(colors))
        ax.add_collection(col)

        # Highlight outlines for observed nucleotide pairs
        if highlight_patches:
            hl_col = PatchCollection(
                highlight_patches, facecolors='none',
                edgecolors='black', linewidths=0.8, zorder=2
            )
            ax.add_collection(hl_col)

        # ── Position labels along the top ──────────────────────────────────────
        for k in range(L):
            ax.text(k * 2, 1.4, str(k),
                    ha='center', va='bottom', fontsize=8, color='#333333')
            # Show the observed nucleotide letter
            ax.text(k * 2, 0.9, NUCLEOTIDES[obs[k]],
                    ha='center', va='bottom', fontsize=7,
                    color='#666666', style='italic')

        # ── Nucleotide legend (inner cell layout) ──────────────────────────────
        legend_x = (L - 1) * 2 + 2.5
        legend_y = 0.0
        ax.text(legend_x, legend_y + 1.2, 'Inner cell\nlayout',
                ha='center', fontsize=7, color='#444444')
        for a in range(4):
            for b in range(4):
                da, db = a - 1.5, b - 1.5
                lx = legend_x + (da + db) * inner_half * 1.8
                ly = legend_y + (db - da) * inner_half * 1.8
                ax.text(lx, ly, f'{NUCLEOTIDES[a]}/{NUCLEOTIDES[b]}',
                        ha='center', va='center', fontsize=4.5, color='#222222')

        fig.colorbar(col, ax=ax, label=colorbar_label, fraction=0.025, pad=0.04)

        margin = 1.8
        ax.set_xlim(-margin, (L - 1) * 2 + margin + 3.5)
        ax.set_ylim(-(L - 1) - margin, 2.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, pad=20, fontsize=13)
        plt.tight_layout()
        return fig, ax


    # ── Example ─────────────────────────────────────────────────────────────────
    np.random.seed(7)
    L = 6
    # Random one-hot sequence
    indices = np.random.randint(0, 4, size=L)
    one_hot = np.eye(4)[indices]  # shape (L, 4)

    # Random interaction tensor (symmetric)
    Z2 = np.random.randn(L, L, 4, 4)
    for i in range(L):
        for j in range(L):
            sym = (Z2[i, j] + Z2[j, i].T) / 2
            Z2[i, j] = sym
            Z2[j, i] = sym.T

    seq_str = ''.join(NUCLEOTIDES[idx] for idx in indices)
    fig, ax = plot_genomic_interaction_heatmap(
        one_hot, Z2,
        title=f'Genomic Interaction Matrix  |  seq: {seq_str}',
        colorbar_label='Interaction Strength',
        triangle='lower',
        cmap='RdBu_r',
        figsize=(13, 10),
    )
    ax
    return


if __name__ == "__main__":
    app.run()
