import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    def make_rgba(H, seq_len, alphabet=["A", "G", "T", "C"]):
        """Convert Hessian to RGBA image with nucleotide color blending."""
        n_nuc = len(alphabet)
        N = seq_len * n_nuc

        NUC_COLORS = {
            "A": "#2ECC71",
            "G": "#E67E22",
            "T": "#E74C3C",
            "C": "#3498DB",
        }

        H_abs = np.abs(H)
        H_norm = H_abs / H_abs.max()
        rgba = np.ones((N, N, 4))

        for i in range(N):
            for j in range(N):
                nuc_i = alphabet[i % n_nuc]
                nuc_j = alphabet[j % n_nuc]
                c_i = np.array(mcolors.to_rgb(NUC_COLORS[nuc_i]))
                c_j = np.array(mcolors.to_rgb(NUC_COLORS[nuc_j]))
                blended = 0.5 * c_i + 0.5 * c_j
                alpha = H_norm[i, j] ** 0.5
                rgba[i, j, :3] = alpha * blended + (1 - alpha) * 1.0
                rgba[i, j, 3] = 1.0

        return rgba

    def add_axes_labels(ax, seq_len, alphabet, is_reordered, NUC_COLORS):
        """Add colored tick labels and position separators."""
        n_nuc = len(alphabet)
        N = seq_len * n_nuc

        if is_reordered:
            # Position-major: A0 G0 T0 C0 | A1 G1 T1 C1 ...
            tick_labels = [f"{alphabet[i % n_nuc]}{i // n_nuc}" for i in range(N)]
            tick_colors = [NUC_COLORS[alphabet[i % n_nuc]] for i in range(N)]
            for pos in range(1, seq_len):
                ax.axvline(pos * n_nuc - 0.5, color="black", linewidth=1.2)
                ax.axhline(pos * n_nuc - 0.5, color="black", linewidth=1.2)
        else:
            # Nucleotide-major: A0 A1 A2 ... | G0 G1 G2 ...
            tick_labels = [f"{alphabet[i // seq_len]}{i % seq_len}" for i in range(N)]
            tick_colors = [NUC_COLORS[alphabet[i // seq_len]] for i in range(N)]
            for nuc in range(1, n_nuc):
                ax.axvline(nuc * seq_len - 0.5, color="black", linewidth=1.2)
                ax.axhline(nuc * seq_len - 0.5, color="black", linewidth=1.2)

        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)

        for tick, color in zip(ax.get_xticklabels(), tick_colors):
            tick.set_color(color)
            tick.set_fontweight("bold")
        for tick, color in zip(ax.get_yticklabels(), tick_colors):
            tick.set_color(color)
            tick.set_fontweight("bold")

    def plot_hessian_before_after(H_nuc_major, seq_len, alphabet=["A", "G", "T", "C"]):
        """
        Plot Hessian before and after reordering side by side.

        H_nuc_major: Hessian in nucleotide-major layout
                     index = nuc * seq_len + pos
                     i.e. [A0, A1, ..., AN, G0, G1, ..., T0, ..., C0, ...]
        """
        n_nuc = len(alphabet)
        N = seq_len * n_nuc

        NUC_COLORS = {
            "A": "#2ECC71",
            "G": "#E67E22",
            "T": "#E74C3C",
            "C": "#3498DB",
        }

        # Reorder: nuc-major -> pos-major
        idx = np.array(
            [nuc * seq_len + pos for pos in range(seq_len) for nuc in range(n_nuc)]
        )
        H_pos_major = H_nuc_major[np.ix_(idx, idx)]

        # Build RGBA for both — but use the SAME normalization for fair comparison
        H_abs_max = max(np.abs(H_nuc_major).max(), np.abs(H_pos_major).max())

        def make_rgba_shared_norm(H):
            H_norm = np.abs(H) / H_abs_max
            rgba = np.ones((N, N, 4))
            for i in range(N):
                for j in range(N):
                    # In nuc-major: row nuc = i // seq_len, col nuc = j // seq_len
                    # In pos-major: row nuc = i % n_nuc,   col nuc = j % n_nuc
                    # We pass nuc indices explicitly via closure
                    nuc_i_nm = alphabet[i // seq_len]
                    nuc_j_nm = alphabet[j // seq_len]
                    nuc_i_pm = alphabet[i % n_nuc]
                    nuc_j_pm = alphabet[j % n_nuc]
                    return rgba  # placeholder — see below

        def build_rgba(H, row_nuc_fn, col_nuc_fn):
            H_norm = np.abs(H) / H_abs_max
            rgba = np.ones((N, N, 4))
            for i in range(N):
                for j in range(N):
                    c_i = np.array(mcolors.to_rgb(NUC_COLORS[row_nuc_fn(i)]))
                    c_j = np.array(mcolors.to_rgb(NUC_COLORS[col_nuc_fn(j)]))
                    blended = 0.5 * c_i + 0.5 * c_j
                    alpha = H_norm[i, j] ** 0.5
                    rgba[i, j, :3] = alpha * blended + (1 - alpha) * 1.0
            return rgba

        # Nuc-major: index layout is nuc * seq_len + pos -> nuc = i // seq_len
        rgba_before = build_rgba(
            H_nuc_major,
            row_nuc_fn=lambda i: alphabet[i // seq_len],
            col_nuc_fn=lambda j: alphabet[j // seq_len],
        )
        # Pos-major: index layout is pos * n_nuc + nuc -> nuc = i % n_nuc
        rgba_after = build_rgba(
            H_pos_major,
            row_nuc_fn=lambda i: alphabet[i % n_nuc],
            col_nuc_fn=lambda j: alphabet[j % n_nuc],
        )

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        for ax, rgba, is_reordered, title in zip(
            axes,
            [rgba_before, rgba_after],
            [False, True],
            [
                "Before reordering  (nucleotide-major)\n[A0…AN | G0…GN | T0…TN | C0…CN]",
                "After reordering  (position-major)\n[A0 G0 T0 C0 | A1 G1 T1 C1 | …]",
            ],
        ):
            ax.imshow(rgba, aspect="auto", interpolation="nearest")
            add_axes_labels(ax, seq_len, alphabet, is_reordered, NUC_COLORS)
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel("Sequence position × nucleotide")
            ax.set_ylabel("Sequence position × nucleotide")

        # Shared legend
        legend_patches = [
            mpatches.Patch(color=NUC_COLORS[nuc], label=nuc) for nuc in alphabet
        ]
        axes[1].legend(
            handles=legend_patches,
            loc="upper right",
            bbox_to_anchor=(1.13, 1),
            framealpha=0.9,
            title="Nucleotide",
        )

        fig.suptitle(
            "Hessian of one-hot encoded sequence\n"
            "Color = blend of interacting nucleotides  |  Brightness = |H_ij|",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        return fig

    # --- Demo ---
    seq_len = 8
    n_nuc = 4
    N = seq_len * n_nuc
    np.random.seed(42)

    # Build Hessian in nucleotide-major layout: index = nuc * seq_len + pos
    H_raw = np.random.randn(N, N) * 0.3
    H_raw = H_raw @ H_raw.T  # symmetric PSD

    # Boost within-nucleotide blocks (same nuc, different positions)
    for nuc in range(n_nuc):
        sl = slice(nuc * seq_len, (nuc + 1) * seq_len)
        H_raw[sl, sl] *= 3.0

    # Boost a cross-position, cross-nucleotide interaction (A at pos 0 <-> C at pos 5)
    H_raw[0, n_nuc * seq_len - 3] *= 8.0
    H_raw[n_nuc * seq_len - 3, 0] *= 8.0

    fig = plot_hessian_before_after(H_raw, seq_len)
    # plt.savefig('hessian_before_after.png', dpi=150, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _():
    import torch

    # --- Build a fake one-hot sequence (50 positions, 4 nucleotides) ---
    torch.manual_seed(42)
    seq_indices = torch.randint(0, 4, (50,))  # which nucleotide is hot per position
    one_hot = torch.zeros(50, 4)
    one_hot[torch.arange(50), seq_indices] = 1.0
    one_hot = one_hot.bool()

    # --- Build a fake Hessian with known values ---
    # Shape: [1, 50, 4, 1, 50, 4]
    # We'll fill it so hess[0, i, a, 0, j, b] = i*100 + a*10 + j + b*0.1
    # This makes it easy to verify we grabbed the right entries
    hess = torch.zeros(1, 50, 4, 1, 50, 4)
    for i in range(50):
        for a in range(4):
            for j in range(50):
                for b in range(4):
                    hess[0, i, a, 0, j, b] = i * 100 + a * 10 + j + b * 0.1

    # --- Your indexing ---
    idx = one_hot  # [50, 4]
    flat_idx = idx.nonzero(as_tuple=False)[:, 1]  # [50] — hot column per row

    h = hess[0]  # [50, 4, 1, 50, 4]
    h = h[idx]  # [50, 1, 50, 4]
    h = h[:, 0, :, :]  # [50, 50, 4]
    h = h[:, torch.arange(50), flat_idx]  # [50, 50]

    print("h shape:", h.shape)  # should be [50, 50]

    # --- Verify against ground truth ---
    # Expected: h[i, j] = i*100 + seq_indices[i]*10 + j + seq_indices[j]*0.1
    errors = 0
    i_idx = torch.arange(50)
    expected = (
        i_idx.unsqueeze(1) * 100  # i*100,  shape [50,1]
        + seq_indices.unsqueeze(1) * 10  # a*10,   shape [50,1]
        + i_idx.unsqueeze(0)  # j,      shape [1,50]
        + seq_indices.unsqueeze(0) * 0.1  # b*0.1,  shape [1,50]
    )  # broadcasts to [50, 50]

    if torch.allclose(h, expected, atol=1e-3):
        print("✓ All entries match — indexing is correct!")
    else:
        diff = (h - expected).abs()
        print(f"✗ Max error: {diff.max().item():.6f} at {diff.argmax().item()}")

    return


if __name__ == "__main__":
    app.run()
