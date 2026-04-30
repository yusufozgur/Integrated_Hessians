optimize_for_duplicate_interpolation_values = (True,)

(batch_size,)


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
    batch_size=None,
) -> Tuple[
    Annotated[Tensor, "batch_size *input_shape"],  # interaction attributions
    Annotated[Tensor, "batch_size"],  # deltas
]:
    """Computes integrated hessians for feature interaction attributions.

    The function is device agnostic, only requirement is provided tensors should be on the same device.

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
    batch_size : int, optional
        The number of samples to process in parallel during the vectorized
        Hessian and Jacobian computations via ``torch.vmap``. If ``None``,
        processes the entire input batch at once. Adjusting this can help
        manage memory overhead.

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
