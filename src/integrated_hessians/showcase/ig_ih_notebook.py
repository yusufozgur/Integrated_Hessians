import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import torch
    from captum.attr import IntegratedGradients
    from path_explain import PathExplainerTorch
    from integrated_hessians.showcase import (
        surface_from_function,
        elev_slider,
        azim_slider,
        slider_input_x,
        slider_input_y,
        slider_baseline_x,
        slider_baseline_y,
        sample_x_range,
        sample_y_range,
    )
    from integrated_hessians import get_integrated_hessians
    import plotly.express as px

    return (
        IntegratedGradients,
        PathExplainerTorch,
        azim_slider,
        elev_slider,
        get_integrated_hessians,
        np,
        plt,
        px,
        slider_baseline_x,
        slider_baseline_y,
        slider_input_x,
        slider_input_y,
        surface_from_function,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a differentiable surface that behaves similarly to XOR.
    """)
    return


@app.function
def f(X, Y):
    return X + Y - 2 * X * Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gradient $\nabla f(x, y)$ is:
    $$\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) = (1 - 2y, 1 - 2x)$$

    The Hessian matrix $H(f) = \nabla^2 f$ is:
    $$
    H(f) = \begin{bmatrix}
    \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
    \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
    \end{bmatrix} = \begin{bmatrix}
    0 & -2 \\
    -2 & 0
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _(
    azim_slider,
    elev_slider,
    f_baseline,
    f_input,
    mo,
    slider_baseline_x,
    slider_baseline_y,
    slider_input_x,
    slider_input_y,
):
    # Configurations for the notebook
    mo.vstack(
        [
            elev_slider,
            azim_slider,
            slider_input_x,
            slider_input_y,
            f"f(input)={f_input}",
            slider_baseline_x,
            slider_baseline_y,
            f"f(baseline)={f_baseline}",
        ]
    )
    return


@app.cell
def _(
    np,
    slider_baseline_x,
    slider_baseline_y,
    slider_input_x,
    slider_input_y,
):
    # aliasing the input values
    input_x = slider_input_x.value
    input_y = slider_input_y.value
    baseline_x = slider_baseline_x.value
    baseline_y = slider_baseline_y.value
    f_input = f(X=input_x, Y=input_y)
    f_baseline = f(X=baseline_x, Y=baseline_y)

    baseline = 20
    baseline_to_input_path_x = np.linspace(baseline_x, input_x, 50)
    baseline_to_input_path_y = np.linspace(baseline_y, input_y, 50)
    baseline_to_input_path_f = f(baseline_to_input_path_x, baseline_to_input_path_y)
    return (
        baseline_to_input_path_f,
        baseline_to_input_path_x,
        baseline_to_input_path_y,
        baseline_x,
        baseline_y,
        f_baseline,
        f_input,
        input_x,
        input_y,
    )


@app.cell
def _(
    azim_slider,
    baseline_x,
    baseline_y,
    elev_slider,
    f_baseline,
    f_input,
    input_x,
    input_y,
    surface_from_function,
):
    # Show surface and, baseline and input points
    show_surface_fig, show_surface_ax = surface_from_function(
        f, elev_slider.value, azim_slider.value
    )
    show_surface_ax.scatter(
        [input_x], [input_y], [f_input], color="red", s=50, label=f"Input", zorder=5
    )
    show_surface_ax.scatter(
        [baseline_x],
        [baseline_y],
        [f_baseline],
        color="blue",
        s=50,
        label=f"Baseline",
        zorder=5,
    )
    show_surface_ax.set_title(r"XOR-like surface $f(x, y) = x + y - 2xy$")
    show_surface_ax.set_zlabel("f(x, y)")
    show_surface_ax.computed_zorder = False  # to make balls show up
    show_surface_ax.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Integrated Gradients
    """)
    return


@app.cell
def _(
    baseline_to_input_path_x,
    baseline_to_input_path_y,
    surface_from_function,
):
    # Show interpolations in 2d overview
    view_path_fig, view_path_ax = surface_from_function(f, 90, -90)
    view_path_ax.set_zticklabels([])
    view_path_z = f(baseline_to_input_path_x, baseline_to_input_path_y) + 0.8
    view_path_ax.scatter(
        baseline_to_input_path_x,
        baseline_to_input_path_y,
        view_path_z,
        color="red",
        s=20,
        label=f"Point",
        zorder=5,
    )
    view_path_ax.set_title("The path between baseline and the input")
    view_path_fig
    return (view_path_z,)


@app.cell
def _(
    azim_slider,
    baseline_to_input_path_f,
    baseline_to_input_path_x,
    baseline_to_input_path_y,
    elev_slider,
    surface_from_function,
):
    # Show interpolations in 3d view
    view_path_integ_fig, view_path_integ_ax = surface_from_function(
        f, elev_slider.value, azim_slider.value
    )
    view_path_integ_sampling = 20
    view_path_integ_ax.scatter(
        baseline_to_input_path_x,
        baseline_to_input_path_y,
        baseline_to_input_path_f,
        color="red",
        s=20,
        label=f"Point",
        zorder=5,
    )
    view_path_integ_ax.computed_zorder = False  # to make balls show up
    view_path_integ_ax.set_title(
        "The approximation for the path integral between baseline and the input"
    )
    view_path_integ_ax.set_zlabel("f(x, y)")
    # view_path_integ_ax.set_proj_type('ortho')
    view_path_integ_fig
    return


@app.cell
def _():
    # mo.vstack([
    #     show_surface_fig,
    #     view_path_fig,
    #     view_path_integ_fig,
    #     mo.hstack([slice_2d_x_with_path_fig, slice_2d_y_with_path_fig]),

    # ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Integrated gradients integrates a path integral. This is approximated by sampling the path.
    """)
    return


@app.cell
def _(baseline_to_input_path_x, baseline_to_input_path_y, torch):
    path_tensor = torch.stack(
        [torch.tensor(baseline_to_input_path_x), torch.tensor(baseline_to_input_path_y)]
    ).transpose(1, 0)
    path_tensor.requires_grad = True

    f_w_point_tensor = lambda x: f(x[0], x[1])
    points_f = torch.vmap(f_w_point_tensor)(path_tensor)
    points_grad = torch.vmap(torch.func.grad(f_w_point_tensor))(path_tensor)

    points_f, points_grad
    return points_f, points_grad


@app.cell
def _(baseline_to_input_path_x, np, plt, points_f, points_grad):
    slice_2d_x_with_path_fig, slice_2d_x_with_path_ax = plt.subplots(figsize=(10, 6))
    slice_2d_x_with_path__x = baseline_to_input_path_x
    slice_2d_x_with_path__y1 = points_f.detach().numpy()
    slice_2d_x_with_path__y2 = points_grad[:, 0].detach().numpy()
    slice_2d_x_with_path__dx = np.ones_like(slice_2d_x_with_path__x)
    slice_2d_x_with_path__dy = slice_2d_x_with_path__y2
    slice_2d_x_with_path__magnitude = np.sqrt(
        slice_2d_x_with_path__dx**2 + slice_2d_x_with_path__dy**2
    )
    slice_2d_x_with_path__dx_norm = (
        slice_2d_x_with_path__dx / slice_2d_x_with_path__magnitude
    )
    slice_2d_x_with_path__dy_norm = (
        slice_2d_x_with_path__dy / slice_2d_x_with_path__magnitude
    )

    slice_2d_x_with_path_ax.plot(
        slice_2d_x_with_path__x, slice_2d_x_with_path__y1, color="red", label="Path"
    )
    slice_2d_x_with_path_ax.plot(
        slice_2d_x_with_path__x,
        slice_2d_x_with_path__y2,
        color="green",
        label="Gradient",
    )
    slice_2d_x_with_path__q = slice_2d_x_with_path_ax.quiver(
        slice_2d_x_with_path__x[::5],
        slice_2d_x_with_path__y1[::5],
        slice_2d_x_with_path__dx_norm[::5],
        slice_2d_x_with_path__dy_norm[::5],
        scale=20,
        width=0.003,
        color="black",
    )
    slice_2d_x_with_path_ax.plot(
        [],
        [],
        color="black",
        marker=r"$\rightarrow$",
        markersize=15,
        label="Gradient Arrow",
        linestyle="None",
    )
    # slice_2d_x_with_path_ax.quiverkey(q, X=0.85, Y=0.05, U=1, label="Gradient Arrow", labelpos='E')
    slice_2d_x_with_path_ax.set_title("Projected onto x: Path and the gradients")
    slice_2d_x_with_path_ax.set_xlabel("x")
    slice_2d_x_with_path_ax.set_ylabel("f")
    slice_2d_x_with_path_ax.grid(
        True, color="gray", linestyle="-", linewidth=0.5, alpha=0.7
    )
    slice_2d_x_with_path_ax.legend()
    return (slice_2d_x_with_path__y2,)


@app.cell
def _(
    baseline_to_input_path_y,
    np,
    plt,
    points_f,
    points_grad,
    slice_2d_x_with_path__y2,
):
    slice_2d_y_with_path_fig, slice_2d_y_with_path_ax = plt.subplots(figsize=(10, 6))
    slice_2d_y_with_path__x = baseline_to_input_path_y
    slice_2d_y_with_path__y1 = points_f.detach().numpy()
    slice_2d_y_with_path__y2 = points_grad[:, 1].detach().numpy()
    slice_2d_y_with_path__dx = np.ones_like(slice_2d_y_with_path__x)
    slice_2d_y_with_path__dy = slice_2d_x_with_path__y2
    slice_2d_y_with_path__magnitude = np.sqrt(
        slice_2d_y_with_path__dx**2 + slice_2d_y_with_path__dy**2
    )
    slice_2d_y_with_path__dx_norm = (
        slice_2d_y_with_path__dx / slice_2d_y_with_path__magnitude
    )
    slice_2d_y_with_path__dy_norm = (
        slice_2d_y_with_path__dy / slice_2d_y_with_path__magnitude
    )

    slice_2d_y_with_path_ax.plot(
        slice_2d_y_with_path__x, slice_2d_y_with_path__y1, color="red", label="Path"
    )
    slice_2d_y_with_path_ax.plot(
        slice_2d_y_with_path__x,
        slice_2d_y_with_path__y2,
        color="green",
        label="Gradient",
    )
    slice_2d_y_with_path__q = slice_2d_y_with_path_ax.quiver(
        slice_2d_y_with_path__x[::5],
        slice_2d_y_with_path__y1[::5],
        slice_2d_y_with_path__dx_norm[::5],
        slice_2d_y_with_path__dy_norm[::5],
        scale=20,
        width=0.003,
        color="black",
    )
    slice_2d_y_with_path_ax.plot(
        [],
        [],
        color="black",
        marker=r"$\rightarrow$",
        markersize=15,
        label="Gradient Arrow",
        linestyle="None",
    )
    # slice_2d_x_with_path_ax.quiverkey(q, X=0.85, Y=0.05, U=1, label="Gradient Arrow", labelpos='E')
    slice_2d_y_with_path_ax.set_title("Projected onto x: Path and the gradients")
    slice_2d_y_with_path_ax.set_xlabel("x")
    slice_2d_y_with_path_ax.set_ylabel("f")
    slice_2d_y_with_path_ax.grid(
        True, color="gray", linestyle="-", linewidth=0.5, alpha=0.7
    )
    slice_2d_y_with_path_ax.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Integrated gradients takes an integral along this path of gradients. You will see that IG results for this function is 0, and its apparent the integral of the gradient path will sum up to 0 in the plot too.
    """)
    return


@app.function
def f_batched(t):
    return t[:, 0] + t[:, 1] - 2 * t[:, 0] * t[:, 1]


@app.cell
def _(IntegratedGradients, baseline_x, baseline_y, input_x, input_y, torch):
    # Captum
    ig = IntegratedGradients(f_batched)
    input_tensor = torch.tensor(
        [input_x, input_y], dtype=torch.float64, requires_grad=True
    ).unsqueeze(0)
    baseline_tensor = torch.tensor(
        [baseline_x, baseline_y], dtype=torch.float64, requires_grad=True
    ).unsqueeze(0)
    input_tensor.shape
    return baseline_tensor, ig, input_tensor


@app.cell
def _(baseline_tensor, ig, input_tensor):
    attributions, delta = ig.attribute(
        input_tensor, baseline_tensor, n_steps=50, return_convergence_delta=True
    )
    f"Captum Attributions: {attributions.detach().numpy()} with delta: {float(delta.detach().numpy()[0]): .2E}"
    return (attributions,)


@app.cell
def _(np, torch):
    # our implementation

    # I think the biggest difference in our implementation could be to make the linear path its own function, and take gradients with respect to that
    # Usually the formula for integrated gradients is written in vector form as:
    #   IG = (x-x')*sum_k=1^m=50{dF(x' + k/m * (x-x'))/dx * 1/m}
    # However, as the inside of the function is the interpolation of the linear path, we can express the formula in a simpler form by taking the interpolation into a seperate variable
    #   L(a) = x'+a*(x-x') where a/alpha is the interpolation coefficient, which was k/m from before
    #   a in np.linspace(0, 1, 50)
    #   let G(a) = F(L(a))
    #   IG = (x-x')*sum_a{dG(a)/da * 1/m}
    #   we do not need to explicitly calculate the derivative of G(a) using the chain rule, as autograd should handle that

    def path_ig(F, input, baseline, target: int, n_steps=50, retain_graph=False):
        alphas = np.linspace(0, 1, n_steps)

        def L(a):
            return baseline + a * (input - baseline)

        rimann_sum = torch.zeros_like(input)
        for a in alphas:
            interpolation = L(a)
            out = F(interpolation)
            out = out[target]

            grad_tuple = torch.autograd.grad(outputs=out, inputs=interpolation)
            grad = grad_tuple[0]

            rimann_sum += grad

        rimann_sum *= 1 / n_steps  # we apply 1/m outside the loop as its equivalent

        ig_result = (input - baseline) * rimann_sum

        delta = ig_result.sum() - (F(input) - F(baseline))

        return ig_result, delta

    return (path_ig,)


@app.cell
def _(baseline_tensor, input_tensor, path_ig):
    path_ig_attr, path_ig_delta = path_ig(
        f_batched, input_tensor, baseline_tensor, target=0
    )
    f"Path IG(Our) attributions: {path_ig_attr.detach().numpy()} with delta: {float(path_ig_delta.detach().numpy()[0]): .2E}"
    return (path_ig_attr,)


@app.cell
def _(attributions, np, path_ig_attr, plt):
    attr_plot_data = np.concat(
        [attributions.detach().numpy(), path_ig_attr.detach().numpy()]
    )
    attr_plot_fig, attr_plot_ax = plt.subplots(figsize=(3, 3))
    attr_plot_ax.imshow(attr_plot_data, cmap="Blues", vmin=-1, vmax=1)
    for i in range(2):
        for j in range(2):
            attr_plot_ax.text(
                j,
                i,
                f"{attr_plot_data[i, j]:.2E}",
                ha="center",
                va="center",
                fontsize=14,
            )
    attr_plot_ax.set_xticks([0, 1])
    attr_plot_ax.set_xticklabels(["X", "Y"])
    attr_plot_ax.set_yticks([0, 1])
    attr_plot_ax.set_yticklabels(["Captum", "Our implementation"])
    attr_plot_ax.set_title("Integrated Gradients Attributions")
    attr_plot_ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
    attr_plot_ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    attr_plot_ax.grid(which="minor", color="black", linewidth=1)
    attr_plot_ax.tick_params(which="minor", length=0)
    attr_plot_ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Integrated Hessians
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Integrated Gradients does not inform us of feature interactions that are affecting the prediction.

    One can apply IG on feature attributions acquired from IG to figure out how each a feature may change the attribution of another feature.

    IH(x,i,j) = IG(IG(x,i),j)

    This is a double path integral.

    In IG, you would sample a path from baseline to input.

    In IH, you will trace multiple such path
    """)
    return


@app.cell
def _(
    baseline_to_input_path_x,
    baseline_to_input_path_y,
    surface_from_function,
    view_path_z,
):
    view_path_ih_top_fig, view_path_ih_top_ax = surface_from_function(f, 90, -90)
    view_path_ih_top_ax.set_zticklabels([])
    view_path_ih_z = f(baseline_to_input_path_x, baseline_to_input_path_y) + 0.8
    view_path_ih_top_ax.scatter(
        baseline_to_input_path_x,
        baseline_to_input_path_y,
        view_path_z,
        color="red",
        s=20,
        label=f"Point",
        zorder=5,
    )
    view_path_ih_top_ax.set_title("The path between baseline and the input")
    view_path_ih_top_ax
    return


@app.cell
def _(azim_slider, elev_slider, surface_from_function):
    view_path_ih_fig, view_path_ih_ax = surface_from_function(
        f, elev_slider.value, azim_slider.value
    )
    view_path_ih_ax
    return


@app.cell
def _():
    # in 2d or 3d, draw paths
    return


@app.cell
def _(PathExplainerTorch, baseline_tensor, input_tensor):
    # path_explain
    def f_vectorized2(t):
        return (t[:, 0] + t[:, 1] - 2 * t[:, 0] * t[:, 1]).unsqueeze(-1)
    exp = PathExplainerTorch(f_vectorized2)
    path_explain_interactions = exp.interactions(
        input_tensor, baseline_tensor, use_expectation=False, num_samples=3
    )
    return (path_explain_interactions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Janizek et al IH implementation:
    """)
    return


@app.cell
def _(path_explain_interactions):
    path_explain_interactions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fix for non diag: Middle riemann sum converges much faster

    Fix for diag: include the additional term for i=j case
    """)
    return


@app.cell
def _(baseline_tensor, get_integrated_hessians, input_tensor):
    our_ih_attr, our_ih_delta = get_integrated_hessians(
        model=f_batched,
        inputs=input_tensor,
        baselines=baseline_tensor,
        target=None,
        approximation_steps=3,
    )
    return our_ih_attr, our_ih_delta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our IH implementation:
    """)
    return


@app.cell
def _(our_ih_attr):
    our_ih_attr
    return


@app.cell
def _(our_ih_delta):
    our_ih_delta
    return


@app.cell
def _(our_ih_attr, path_explain_interactions, px, torch):
    z = torch.cat([path_explain_interactions.detach(), our_ih_attr.detach()]).numpy().transpose(1,0,2).reshape(2,4)
    fig = px.imshow(z, text_auto=True)
    fig
    return


if __name__ == "__main__":
    app.run()
