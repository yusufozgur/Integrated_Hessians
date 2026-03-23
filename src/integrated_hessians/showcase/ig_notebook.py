import marimo

__generated_with = "0.20.4"
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
    from integrated_hessians.showcase import surface_from_function, elev_slider, azim_slider, slider_input_x, slider_input_y, slider_baseline_x, slider_baseline_y, sample_x_range, sample_y_range

    return (
        IntegratedGradients,
        PathExplainerTorch,
        azim_slider,
        elev_slider,
        np,
        plt,
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
    mo.vstack([
        elev_slider, 
        azim_slider, 
        slider_input_x, 
        slider_input_y, 
        f"f(input)={f_input}",
        slider_baseline_x,
        slider_baseline_y,
        f"f(baseline)={f_baseline}",
    ])
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
    baseline_to_input_path_x = np.linspace(baseline_x,input_x,50)
    baseline_to_input_path_y = np.linspace(baseline_y,input_y,50)
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
    show_surface_fig, show_surface_ax = surface_from_function(f, elev_slider.value, azim_slider.value)
    show_surface_ax.scatter([input_x], [input_y], [f_input], color='red', s=50, label=f'Input', zorder=5)
    show_surface_ax.scatter([baseline_x], [baseline_y], [f_baseline], color='blue', s=50, label=f'Baseline', zorder=5)
    show_surface_ax.set_title(r'XOR-like surface $f(x, y) = x + y - 2xy$')
    show_surface_ax.set_zlabel('f(x, y)')
    show_surface_ax.computed_zorder = False # to make balls show up
    show_surface_ax.legend()
    return


@app.cell
def _(
    baseline_to_input_path_x,
    baseline_to_input_path_y,
    surface_from_function,
):
    # Show interpolations in 2d overview
    view_path_fig, view_path_ax = surface_from_function(f,90,-90)
    view_path_ax.set_zticklabels([])
    view_path_z = f(baseline_to_input_path_x, baseline_to_input_path_y) + .8
    view_path_ax.scatter(baseline_to_input_path_x, baseline_to_input_path_y, view_path_z, color='red', s=20, label=f'Point', zorder=5)
    view_path_ax.set_title("The path between baseline and the input")
    view_path_fig
    return


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
    view_path_integ_fig, view_path_integ_ax = surface_from_function(f, elev_slider.value, azim_slider.value)
    view_path_integ_sampling = 20
    view_path_integ_ax.scatter(baseline_to_input_path_x, baseline_to_input_path_y, baseline_to_input_path_f, color='red', s=20, label=f'Point', zorder=5)
    view_path_integ_ax.computed_zorder = False # to make balls show up
    view_path_integ_ax.set_title("The approximation for the path integral between baseline and the input")
    view_path_integ_ax.set_zlabel('f(x, y)')
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
def _():
    # path_tensor_x = torch.tensor(baseline_to_input_path_x, requires_grad=True)
    # path_tensor_y = torch.tensor(baseline_to_input_path_y, requires_grad=True)

    return


@app.cell
def _(baseline_to_input_path_x, baseline_to_input_path_y, torch):
    path_tensor = torch.stack([
        torch.tensor(baseline_to_input_path_x),
        torch.tensor(baseline_to_input_path_y)
    ]).transpose(1,0)
    path_tensor.requires_grad = True

    f_w_point_tensor = lambda x: f(x[0], x[1])
    points_f = torch.vmap(f_w_point_tensor)(path_tensor)
    points_grad = torch.vmap(torch.func.grad(f_w_point_tensor))(path_tensor)

    points_f, points_grad
    return points_f, points_grad


@app.cell
def _(baseline_to_input_path_x, plt, points_f, points_grad):
    slice_2d_x_with_path_fig, slice_2d_x_with_path_ax = plt.subplots(figsize=(6, 4))
    slice_2d_x_with_path_ax.plot(baseline_to_input_path_x, points_f.detach().numpy(), color="red", label="Path")
    slice_2d_x_with_path_ax.plot(baseline_to_input_path_x, points_grad[:,0].detach().numpy(), color="green", label="Gradient")
    slice_2d_x_with_path_ax.set_title("x vs f")
    slice_2d_x_with_path_ax.set_xlabel("x")
    slice_2d_x_with_path_ax.set_ylabel("f")
    slice_2d_x_with_path_ax.legend()
    return


@app.cell
def _(baseline_to_input_path_f, baseline_to_input_path_y, plt, points_grad):
    slice_2d_y_with_path_fig, slice_2d_y_with_path_ax = plt.subplots(figsize=(6, 4))
    slice_2d_y_with_path_ax.plot(baseline_to_input_path_y, baseline_to_input_path_f, color="red", label="Path")
    slice_2d_y_with_path_ax.plot(baseline_to_input_path_y, points_grad[:,1].detach().numpy(), color="green", label="Gradient")
    slice_2d_y_with_path_ax.set_title("y vs f")
    slice_2d_y_with_path_ax.set_xlabel("y")
    slice_2d_y_with_path_ax.set_ylabel("f")
    slice_2d_y_with_path_ax.legend()
    return


@app.function
def f_batched(t):
    return t[:,0] + t[:,1] - 2 * t[:,0] * t[:,1]


@app.cell
def _(IntegratedGradients, baseline_x, baseline_y, input_x, input_y, torch):
    # Captum
    ig = IntegratedGradients(f_batched)
    input_tensor = torch.tensor([input_x,input_y],dtype=torch.float32,requires_grad=True).unsqueeze(0)
    baseline_tensor = torch.tensor([baseline_x,baseline_y],dtype=torch.float32,requires_grad=True).unsqueeze(0)
    input_tensor.shape
    return baseline_tensor, ig, input_tensor


@app.cell
def _(baseline_tensor, ig, input_tensor):
    attributions, delta = ig.attribute(input_tensor, baseline_tensor, n_steps=50, return_convergence_delta=True)
    f"Captum Attributions: {attributions.detach().numpy()} with delta: {float(delta.detach().numpy()[0]): .2E}"
    return


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

    def path_ig(F, input, baseline, target: int, n_steps=50):
        alphas = np.linspace(0, 1, n_steps)
        def L(a):
            return baseline + a * (input - baseline)

        rimann_sum = torch.zeros_like(input)
        for a in alphas:
            interpolation = L(a)
            out = F(interpolation)
            out = out[target]

            grad_tuple = torch.autograd.grad(outputs=out,inputs=interpolation)
            grad = grad_tuple[0]

            rimann_sum += grad

        rimann_sum *= (1/n_steps) # we apply 1/m outside the loop as its equivalent

        ig_result = (input - baseline) * rimann_sum

        delta = (ig_result.sum() - (F(input) - F(baseline)))

        return ig_result, delta


    return (path_ig,)


@app.cell
def _(baseline_tensor, input_tensor, path_ig):
    path_ig_attr, path_ig_delta = path_ig(f_batched,input_tensor,baseline_tensor, target=0)
    f"Path IG(Our) attributions: {path_ig_attr.detach().numpy()} with delta: {float(path_ig_delta.detach().numpy()[0]): .2E}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Integrated Gradients does not inform us of feature interactions that are affecting the prediction.

    One can apply IG on feature attributions acquired from IG to figure out how each a feature may change the attribution of another feature.

    IH(x,i,j) = IG(IG(x,i),j)

    This is a double path integral.
    """)
    return


@app.cell
def _():
    # in 2d or 3d, draw paths
    return


@app.cell
def _(PathExplainerTorch):
    # path_explain
    exp = PathExplainerTorch(f_vectorized2)
    return (exp,)


@app.cell
def _(input_tensor):
    input_tensor
    return


@app.cell
def _(f_vectorized, input_tensor):
    f_vectorized(input_tensor).detach()
    return


@app.function
def f_vectorized2(t):
    return (t[:,0] + t[:,1] - 2 * t[:,0] * t[:,1]).unsqueeze(-1)


@app.cell
def _(baseline_tensor, exp, input_tensor):
    exp.interactions(input_tensor,baseline_tensor, use_expectation=False)
    return


@app.cell
def _(torch):
    # our implementation
    def path_ih(f, input, baseline, n_steps=50):
        def L(a):
            return baseline + a * (input - baseline)

        input = input.reshape(1,2)
        baseline = baseline.reshape(1,2)

        diff = input - baseline

        outer_product = diff.unsqueeze(1)*diff.unsqueeze(2)

        outer_product = outer_product.reshape(1, 4)

        k = n_steps
        m = n_steps

        riem_sum = torch.zeros(1,4)

        for l in range(1, k + 1):
            for p in range(1, m + 1):
                alpha = l / k * p / m

                # cannot handle multi sample inputs right now
                second_order_grad = torch.autograd.functional.hessian(
                    f, L(alpha), strict=True,
                ).reshape(1,4)

                riem_sum += second_order_grad * alpha

        riem_sum = riem_sum * 1 / (k * m) * outer_product

        riem_sum = riem_sum.reshape(1,2,2)
        return riem_sum

    return (path_ih,)


@app.cell
def _(input_tensor):
    input_tensor.shape
    return


@app.cell
def _(baseline_tensor, f_vectorized, input_tensor, path_ih):
    path_ih(f_vectorized, input_tensor, baseline_tensor)
    return


@app.cell
def _(torch):
    tmp = torch.arange(1,5).unsqueeze(0)
    tmp
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
