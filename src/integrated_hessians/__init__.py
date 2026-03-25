from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Annotated, Literal, Tuple, Union
from beartype import beartype
from torch import Tensor
import torch
import torch.nn.functional as F
import jaxtyping as jx
from tqdm import tqdm


@contextmanager
def replace_relu_with_softplus(
    implementation: Literal["janizeketal", "pytorch"] = "janizeketal",
    beta=10,  # chosen due to the Janizek et al paper and their path_explain github repo
):
    """
    This context manager changes all calls to the ReLU with softplus, also called monkey patching. It works for both torch.nn.functional.ReLU calls and torch.nn.ReLU calls.

    use it like the following:
    ```
    model = nn.Sequential(nn.ReLU())
    input = torch.Tensor([-1])
    print("without context", model(input))
    with replace_relu_with_softplus():
        print("with context", model(input))
    print("after cleanup context", model(input))
    ```
    """

    def custom_softplus(input, beta):
        match implementation:
            # softplus implemented by janizek et al
            # this seems to be better as we dont need a thresholding value, so we are differentiable at all ranges, at least what I understood.
            case "janizeketal":
                return (1.0 / beta) * torch.log(
                    1.0 + torch.exp(-torch.abs(beta * input))
                ) + torch.maximum(input, torch.zeros_like(input))
            case "pytorch":
                # this is a different softplus compared to one used by Janizek et al. I decided to go with janizek implementation as that one did not require the threshold value
                threshold = (
                    20.0  # chosen arbitrarily, used for pytorch softplus implementation
                )
                return F.softplus(input, beta=beta, threshold=threshold)
            case _:
                raise ValueError("wrong value for implementation")

    def replacement_softplus(input, inplace=False):
        if inplace:
            raise ValueError(
                "ReLU replacement should not be done where ReLU is called with inplace=True"
            )
        return custom_softplus(input, beta=beta)

    original_relu = F.relu
    try:
        F.relu = replacement_softplus
        yield None  # to conform to the contextmanager convention, otherwise an error raises.
    finally:
        F.relu = original_relu


@jx.jaxtyped(typechecker=beartype)
def get_hessian(
    model: Callable,
    input: jx.Float[Tensor, " batch_size *input_shape"],
    target: int,
) -> Annotated[Tensor, "batch_size *input_shape batch_size *input_shape"]:
    def forward_func(x):
        return model(x)[target]

    with replace_relu_with_softplus():
        hessian_result: jx.Float[
            Tensor, "batch_size *input_shape batch_size *input_shape"
        ] = torch.autograd.functional.hessian(forward_func, input, strict=True)  # type: ignore

    # jaxtyping does not support multiple variadic specifiers (*name,). So, instead, I am checking the output is correct shape via an assert statement
    assert hessian_result.shape == (*input.shape, *input.shape)

    return hessian_result


def _get_interpolation_coefficients(
    mode: Literal["default", "multiplication_table_optimized"],
    approximation_steps: int,
    verbose: bool,
):
    """
    For example, in a multiplication table, there are multiple ways to get 16, 2*8, 4*4 or 8*2. Similarly, we do not need to calculate hessians if the interpolated value will be the same. mode == "multiplication_table_optimized" acccounts for that
    """

    k = approximation_steps
    m = approximation_steps

    assert mode in ["default", "multiplication_table_optimized"], (
        "You must provide a valid value for mode parameter."
    )

    if mode == "default":
        alphabetas: list[float] = []
        weights: list[float] = []
        # Important: we use the middle riemann sum, because it approximates better than right riemann sum. Usage of (l - .5) and (p - .5) is due to the middle riemann sum.
        for l in range(1, k + 1):  # noqa: E741
            beta: float = (l - 0.5) / k  # -.5 is for getting the middle riemann sum
            for p in range(1, m + 1):
                alpha: float = (p - 0.5) / m

                # alphabeta is the combined interpolation coefficient
                alphabeta: float = beta * alpha

                alphabetas.append(alphabeta)
                weights.append(1)
        return alphabetas, weights
    elif mode == "multiplication_table_optimized":
        alphabetas_dict = defaultdict(int)

        for l in range(1, k + 1):  # noqa: E741
            beta: float = (l - 0.5) / k  # -.5 is for getting the middle riemann sum
            for p in range(1, m + 1):
                alpha: float = (p - 0.5) / m

                # alphabeta is a float, hence, I dont wanna use it as dictionary key due to floating point imprecision.
                alphabeta_key = l * p
                alphabetas_dict[alphabeta_key] += 1

        alphabetas = [(l_times_k / k / m) for l_times_k in alphabetas_dict.keys()]
        weights = list(alphabetas_dict.values())

        if verbose:
            print(
                f"multiplication table deduplication reduced calculations by {(1 - len(alphabetas) / approximation_steps**2) * 100}%"
            )

        return alphabetas, weights


@jx.jaxtyped(typechecker=beartype)
def _get_common_term(
    func: Callable[
        [jx.Float[Tensor, " input_shape_flattened "]],
        jx.Float[Tensor, ""],
    ],
    inputs_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
    baselines_flattened: jx.Float[Tensor, "batch_size input_shape_flattened "],
    approximation_steps: int,
    interpolation_coefficients_mode: Literal[
        "default", "multiplication_table_optimized"
    ],
) -> jx.Float[Tensor, "batch_size input_shape_flattened input_shape_flattened"]:
    """
    If i != j, then

    IH = (xi - xi') (xj - xj') sum_l=1^k (sum_p=1^m( l/k*p/m * df(x' + (l/k) (x - x'))/(dxi dxj) 1/k/m ))

    We name term the common term, as it is calculated for both i==j and i!=j cases, the extra term for the i==j case is calculated elsewhere
    """
    batch_shape = inputs_flattened.shape[0]
    flattened_input_shape = inputs_flattened.shape[1]

    # this carries the riemann sum
    second_order_sensitivity = torch.zeros(
        batch_shape, flattened_input_shape, flattened_input_shape
    )

    get_second_order_grad = torch.vmap(torch.func.hessian(func=func))

    alphabetas, weights = _get_interpolation_coefficients(
        mode=interpolation_coefficients_mode,
        approximation_steps=approximation_steps,
        verbose=True,
    )

    for alphabeta, weight in tqdm(zip(alphabetas, weights), total=len(alphabetas)):
        # This is one point on the path integral
        interpolation: jx.Float[Tensor, "batch_size input_shape_flattened "] = (
            baselines_flattened + alphabeta * (inputs_flattened - baselines_flattened)
        )

        second_order_grad: jx.Float[
            Tensor, "batch_size input_shape_flattened input_shape_flattened"
        ] = get_second_order_grad(interpolation)
        # this is important to not get memory overflows
        # as otherwise the accumulated tensor will remember every graph
        second_order_grad = second_order_grad.detach()

        second_order_sensitivity += (
            second_order_grad
            * alphabeta
            * (1 / approximation_steps / approximation_steps)  # 1 / k / m
            * weight
        )

    diff = inputs_flattened - baselines_flattened
    outer_product = diff.unsqueeze(dim=1) * diff.unsqueeze(dim=2)
    common_term = second_order_sensitivity * outer_product
    return common_term


@jx.jaxtyped(typechecker=beartype)
def _get_self_interaction_extra_term(
    func: Callable[
        [jx.Float[Tensor, " input_shape_flattened "]],
        jx.Float[Tensor, ""],
    ],
    inputs_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
    baselines_flattened: jx.Float[Tensor, "batch_size input_shape_flattened "],
    approximation_steps: int,
    interpolation_coefficients_mode: Literal[
        "default", "multiplication_table_optimized"
    ],
) -> jx.Float[Tensor, "batch_size input_shape_flattened"]:
    """
    if i == j, then

    IH = (xi - xi') sum_l=1^k(sum_p=1^m( df(x' + (l/k) (x - x'))/dxi 1/k/m )) + (xi - xi') (xj - xj') sum_l=1^k(sum_p=1^m( l/k*p/m * df(x' + (l/k) (x - x'))/(dxi dxj) 1/k/m ))

    The second term is already calculated at _get_common_term(), here, we calculate  the first term.
    """
    batch_shape = inputs_flattened.shape[0]
    flattened_input_shape = inputs_flattened.shape[1]

    # this carries the riemann sum
    self_interaction_term = torch.zeros(batch_shape, flattened_input_shape)

    get_jacobian = torch.vmap(torch.func.jacrev(func=func))

    for alphabeta, weight in zip(
        *_get_interpolation_coefficients(
            mode=interpolation_coefficients_mode,
            approximation_steps=approximation_steps,
            verbose=False,
        )
    ):
        # This is one point on the path integral
        interpolation: jx.Float[Tensor, "batch_size input_shape_flattened "] = (
            baselines_flattened + alphabeta * (inputs_flattened - baselines_flattened)
        )

        first_order_grad: jx.Float[
            Tensor, "batch_size input_shape_flattened input_shape_flattened"
        ] = get_jacobian(interpolation)

        # this is important to not get memory overflows
        first_order_grad = first_order_grad.detach()

        self_interaction_term += (
            first_order_grad
            * (1 / approximation_steps / approximation_steps)  # 1 / k / m
            * weight
        )

    diff = inputs_flattened - baselines_flattened
    self_interaction_term = diff * self_interaction_term
    return self_interaction_term


@jx.jaxtyped(typechecker=beartype)
def _get_delta(
    func: Callable[
        [jx.Float[Tensor, " batch_size input_shape_flattened "]],
        jx.Float[Tensor, " batch_size "],
    ],
    inputs_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
    baselines_flattened: jx.Float[Tensor, "batch_size input_shape_flattened "],
    interaction_attributions: jx.Float[
        Tensor, "batch_size input_shape_flattened input_shape_flattened"
    ],
) -> jx.Float[Tensor, " batch_size "]:
    f_input: jx.Float[Tensor, " batch_size 1 "] = func(inputs_flattened)
    f_baseline: jx.Float[Tensor, " batch_size 1 "] = func(baselines_flattened)

    out_diff: jx.Float[Tensor, " batch_size 1 "] = f_input - f_baseline

    interaction_attr_sum: jx.Float[Tensor, " batch_size "] = (
        interaction_attributions.sum(dim=-1).sum(dim=-1)
    )

    delta: jx.Float[Tensor, " batch_size "] = out_diff - interaction_attr_sum

    return delta


def get_integrated_hessians(
    model: Callable[
        [jx.Float[Tensor, " batch_size *input_shape "]],
        jx.Float[Tensor, " batch_size *output_shape "],
    ],
    inputs: jx.Float[Tensor, " batch_size *input_shape "],
    baselines: jx.Float[Tensor, " batch_size *input_shape "],
    target: Union[None, int, Tuple[int, ...]],
    approximation_steps=50,
    optimize_for_duplicate_interpolation_values=True,
) -> Tuple[
    Annotated[Tensor, "batch_size *input_shape"],  # interaction attributions
    Annotated[Tensor, "batch_size"],  # deltas
]:
    """Computes integrated hessians for feature interaction attributions.

    Calculates second-order interaction attributions between input features by
    integrating the Hessian of the model output along a path from a baseline to
    the input. This decomposes the model output into pairwise feature interaction
    terms, analogous to how integrated gradients decompose outputs into first-order
    feature attributions.

    ReLU activations are replaced with softplus during computation to ensure
    smooth second-order derivatives.

    Parameters
    ----------
    model : Callable
        A callable that maps batched inputs of shape ``(batch_size, *input_shape)``
        to batched outputs of shape ``(batch_size, *output_shape)``.
    inputs : Float[Tensor, "batch_size *input_shape"]
        Input tensor. Must have at least two dimensions, where the first is the
        batch dimension.
    baselines : Float[Tensor, "batch_size *input_shape"]
        Baseline tensor representing the reference point for the path integral.
        Must match the shape of ``inputs``.
    target : None or int or tuple of int
        Index or indices used to select a scalar output from the model predictions
        via ``output[:, target]``. Pass ``None`` to skip subsetting, in which case
        the model must already return a scalar per batch element.
    approximation_steps : int, optional
        Number of steps used for the Riemann sum approximation of the double path
        integral. Higher values yield more accurate results at greater
        computational cost. Defaults to 50.
    optimize_for_duplicate_interpolation_values : bool, optional
        If True, groups identical interpolation coefficient alpha*beta on the
        integration grid to avoid redundant calculations. This significantly reduces
        the number of Hessian/Jacobian evaluations by up to ~70%.

    Returns
    -------
    interaction_attributions : Tensor of shape (batch_size, *input_shape, *input_shape)
        Pairwise interaction attribution scores. Diagonal entries ``[b, i, i]``
        represent self-interaction (analogous to integrated gradients), while
        off-diagonal entries ``[b, i, j]`` capture the interaction between
        features ``i`` and ``j`` for batch element ``b``.
    deltas : Tensor of shape (batch_size,)
        Approximation error per sample, defined as the difference between the
        actual change in model output ``f(input) - f(baseline)`` and the sum of
        all interaction attributions. Values close to zero indicate a reliable
        approximation.

    Raises
    ------
    AssertionError
        If ``inputs`` has fewer than two dimensions.
    AssertionError
        If ``baselines`` does not match the shape of ``inputs``.
    AssertionError
        If the model output cannot be reduced to a scalar per batch element using
        the provided ``target``.
    """
    assert len(inputs.shape) >= 2, (
        "inputs must have at least two dimensions, one bath dimension and one or more input dimensions"
    )

    batch_shape = inputs.shape[0]
    input_shape = inputs.shape[1:]

    assert baselines.shape == (batch_shape, *input_shape), (
        "Input tensor and baseline tensor must have the same shape"
    )

    # instead of an unknown shape, it is easier to operate on the flattened tensors

    inputs_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"] = (
        inputs.reshape(batch_shape, -1)
    )
    baselines_flattened: jx.Float[Tensor, "batch_size input_shape_flattened "] = (
        baselines.reshape(batch_shape, -1)
    )

    input_shape_flattened = inputs_flattened.shape[1]

    @jx.jaxtyped(typechecker=beartype)
    def func_with_scalar_output(
        x: jx.Float[Tensor, "batch_size input_shape_flattened"],
    ) -> jx.Float[Tensor, " batch_size "]:
        """

        - output a scalar
        - handle flat inputs
        """
        assert len(x.shape) == 2
        assert x.shape[1] == input_shape_flattened
        x_batch_size = x.shape[0]
        # convert to original shape so the user supplied function can accept it
        flat_to_original = x.reshape(x_batch_size, *input_shape)
        output = model(flat_to_original)
        # target does subsetting to get the scalar outputs, so it could be an int or tuple of ints
        if target is not None:
            output = output[:, target]
        assert output.shape == (x_batch_size,), (  # the 1 comes from the unsqueeze
            "Could not reduce output dimension to scalars, the target given for subsetting may be incorrect."
        )
        return output

    @jx.jaxtyped(typechecker=beartype)
    def func_with_scalar_output_and_transient_batch_dimentsion(
        x: jx.Float[Tensor, "input_shape_flattened"],
    ) -> jx.Float[Tensor, ""]:
        """
        Two goals
        - squeeze and unsqueeze to satisfy torch.func.vmap
        - output a scalar
        - handle flat inputs
        """
        # as we use torch.func.vmap, the batch dimension will come here stripped
        assert x.shape == (input_shape_flattened,)
        # add a transient batch dimension
        x_transient_batch = x.unsqueeze(0)

        scalar_output = func_with_scalar_output(x_transient_batch)

        # remove the transient batch dimension
        scalar_output_no_batch_dim = scalar_output.squeeze(0)
        assert scalar_output_no_batch_dim.shape == ()
        return scalar_output_no_batch_dim

    match optimize_for_duplicate_interpolation_values:
        case True:
            interpolation_coefficients_mode = "multiplication_table_optimized"
        case False:
            interpolation_coefficients_mode = "default"

    with replace_relu_with_softplus():
        # used in both i!=j and i==j cases.
        common_term: jx.Float[
            Tensor, "batch_size input_shape_flattened input_shape_flattened"
        ] = _get_common_term(
            func=func_with_scalar_output_and_transient_batch_dimentsion,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            approximation_steps=approximation_steps,
            interpolation_coefficients_mode=interpolation_coefficients_mode,
        )
        # self_interaction_extra_terms are calculated for diagonals(i==j), it exists due to the chain rule of calculus when you take d(IG(xi))/dxi.
        self_interaction_extra_term: jx.Float[
            Tensor, "batch_size input_shape_flattened"
        ] = _get_self_interaction_extra_term(
            func=func_with_scalar_output_and_transient_batch_dimentsion,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            approximation_steps=approximation_steps,
            interpolation_coefficients_mode=interpolation_coefficients_mode,
        )

        # this operation add self interaction term only onto the diagonals of the common term
        interaction_attributions: jx.Float[
            Tensor, "batch_size input_shape_flattened input_shape_flattened"
        ] = common_term + self_interaction_extra_term.diag_embed()

        delta = _get_delta(
            func=func_with_scalar_output,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            interaction_attributions=interaction_attributions,
        )

    interaction_attributions_original_shape = interaction_attributions.reshape(
        batch_shape, *input_shape, *input_shape
    )

    return interaction_attributions_original_shape, delta
