from collections import defaultdict
from typing import Callable, Literal, Optional, Tuple
from beartype import beartype
import jaxtyping as jx
from torch import Size, Tensor
import torch
from tqdm import tqdm
from integrated_hessians._core import (
    PathIntegralStrategy,
    calculated_integrated_hessians,
    integrated_hessians_delta,
    scalar_forward_func_type,
    input_type,
)

flattened_input_type = jx.Float[Tensor, "batch_size input_shape_flattened"]
flattened_hessian_type = jx.Float[
    Tensor, "batch_size input_shape_flattened input_shape_flattened"
]
flattened_forward_func_type = Callable[
    [flattened_input_type],
    jx.Float[Tensor, " batch_size"],
]
riemann_flavors = Literal["midpoint_rule", "trapezoid_rule"]


class RiemannIH(PathIntegralStrategy):
    riemann_flavor: riemann_flavors

    @jx.jaxtyped(typechecker=beartype)
    def __init__(
        self,
        riemann_flavor: riemann_flavors = "midpoint_rule",
        optimize_for_duplicate_interpolation_values: bool = True,
        batch_size: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.riemann_flavor = riemann_flavor
        self.optimize_for_duplicate_interpolation_values = (
            optimize_for_duplicate_interpolation_values
        )

    @jx.jaxtyped(typechecker=beartype)
    def get_integrated_hessians(
        self,
        scalar_forward_function: scalar_forward_func_type,
        inputs: input_type,
        baselines: input_type,
        approximation_steps: int,
    ) -> Tuple[
        calculated_integrated_hessians,  # interaction attributions
        integrated_hessians_delta,  # deltas
    ]:

        # instead of an unknown shape, it is easier to operate on the flattened tensors
        batch_shape = inputs.shape[0]
        input_shape = inputs.shape[1:]
        inputs_flattened: flattened_input_type = inputs.reshape(batch_shape, -1)
        baselines_flattened: flattened_input_type = baselines.reshape(batch_shape, -1)
        input_shape_flattened = inputs_flattened.shape[1]

        flattened_scalar_forward_func: flattened_forward_func_type = (
            self._get_function_with_flat_to_original_shape_conversion(
                f=scalar_forward_function,
                input_shape=input_shape,
                input_shape_flattened=input_shape_flattened,
            )
        )

        @jx.jaxtyped(typechecker=beartype)
        def func_vmap_compatible_transient_batch_dim(
            x: jx.Float[Tensor, "input_shape_flattened"],
        ) -> jx.Float[Tensor, ""]:
            # squeeze and unsqueeze to satisfy torch.func.vmap

            # as we use torch.func.vmap, the batch dimension will come here stripped
            assert x.shape == (input_shape_flattened,)
            # add a transient batch dimension
            x_transient_batch = x.unsqueeze(0)

            scalar_output = flattened_scalar_forward_func(x_transient_batch)

            # remove the transient batch dimension
            scalar_output_no_batch_dim = scalar_output.squeeze(0)
            assert scalar_output_no_batch_dim.shape == ()
            return scalar_output_no_batch_dim

        # used in both i!=j and i==j cases.
        common_term: flattened_hessian_type = self._get_common_term(
            func=func_vmap_compatible_transient_batch_dim,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            approximation_steps=approximation_steps,
            batch_size=self.batch_size,
            riemann_flavor=self.riemann_flavor,
            optimize_for_duplicate_interpolation_values=self.optimize_for_duplicate_interpolation_values,
        ).detach()

        # self_interaction_extra_terms are calculated for diagonals(i==j), it exists due to the chain rule of calculus when you take d(IG(xi))/dxi.
        self_interaction_extra_term: jx.Float[
            Tensor, "batch_size input_shape_flattened"
        ] = self._get_self_interaction_extra_term(
            func=func_vmap_compatible_transient_batch_dim,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            approximation_steps=approximation_steps,
            riemann_flavor=self.riemann_flavor,
            optimize_for_duplicate_interpolation_values=self.optimize_for_duplicate_interpolation_values,
            batch_size=self.batch_size,
        ).detach()

        # this operation add self interaction term only onto the diagonals of the common term
        interaction_attributions: jx.Float[
            Tensor, "batch_size input_shape_flattened input_shape_flattened"
        ] = common_term + self_interaction_extra_term.diag_embed()

        delta = self._get_delta(
            func=flattened_scalar_forward_func,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            interaction_attributions=interaction_attributions,
        ).detach()

        interaction_attributions_original_shape = interaction_attributions.reshape(
            batch_shape, *input_shape, *input_shape
        )

        return interaction_attributions_original_shape, delta

    @staticmethod
    @jx.jaxtyped(typechecker=beartype)
    def _get_function_with_flat_to_original_shape_conversion(
        f: scalar_forward_func_type, input_shape: Size, input_shape_flattened: int
    ) -> flattened_forward_func_type:
        @jx.jaxtyped(typechecker=beartype)
        def flattened_forward_func(
            x: flattened_input_type,
        ) -> jx.Float[Tensor, " batch_size "]:
            assert len(x.shape) == 2
            assert x.shape[1] == input_shape_flattened
            x_batch_size = x.shape[0]
            # convert to original shape so the user supplied function can accept it
            flat_to_original = x.reshape(x_batch_size, *input_shape)
            output = f(flat_to_original)
            assert output.shape == (x_batch_size,), (
                "Could not reduce output dimension to scalars, the target given for subsetting may be incorrect."
            )
            return output

        return flattened_forward_func

    @staticmethod
    @jx.jaxtyped(typechecker=beartype)
    def _get_delta(
        func: flattened_forward_func_type,
        inputs_flattened: flattened_input_type,
        baselines_flattened: flattened_input_type,
        interaction_attributions: flattened_hessian_type,
    ) -> jx.Float[Tensor, " batch_size "]:
        f_input: jx.Float[Tensor, " batch_size "] = func(inputs_flattened)
        f_baseline: jx.Float[Tensor, " batch_size "] = func(baselines_flattened)
        out_diff: jx.Float[Tensor, " batch_size "] = f_input - f_baseline
        interaction_attr_sum: jx.Float[Tensor, " batch_size "] = (
            interaction_attributions.sum(dim=-1).sum(dim=-1)
        )
        delta: jx.Float[Tensor, " batch_size "] = out_diff - interaction_attr_sum
        return delta

    @jx.jaxtyped(typechecker=beartype)
    @staticmethod
    def _get_common_term(
        func: Callable[
            [jx.Float[Tensor, " input_shape_flattened "]],
            jx.Float[Tensor, ""],
        ],
        inputs_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
        baselines_flattened: jx.Float[Tensor, "batch_size input_shape_flattened "],
        approximation_steps: int,
        batch_size: Optional[int],
        riemann_flavor: riemann_flavors,
        optimize_for_duplicate_interpolation_values: bool,
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
            batch_shape,
            flattened_input_shape,
            flattened_input_shape,
            device=inputs_flattened.device,
        )

        get_second_order_grad = torch.vmap(torch.func.hessian(func=func))

        alphabetas, weights = _get_riemann_interpolation_coefficients(
            approximation_steps=approximation_steps,
            verbose=True,
            riemann_flavor=riemann_flavor,
            optimize_for_duplicate_interpolation_values=optimize_for_duplicate_interpolation_values,
        )

        for alphabeta, weight in tqdm(zip(alphabetas, weights), total=len(alphabetas)):
            # This is one point on the path integral
            interpolation: jx.Float[Tensor, "batch_size input_shape_flattened "] = (
                baselines_flattened
                + alphabeta * (inputs_flattened - baselines_flattened)
            )

            if batch_size is None:
                second_order_grad: jx.Float[
                    Tensor, "batch_size input_shape_flattened input_shape_flattened"
                ] = get_second_order_grad(interpolation)
            else:
                batches_second_order_grads = []
                for batch in interpolation.split(batch_size):
                    # this detach is important too in order to not overwhelm gpu memory
                    batches_second_order_grads.append(
                        get_second_order_grad(batch).detach()
                    )

                second_order_grad: jx.Float[
                    Tensor, "batch_size input_shape_flattened input_shape_flattened"
                ] = torch.cat(batches_second_order_grads)

            # this is important to not get memory overflows
            # as otherwise the accumulated tensor will remember every graph
            second_order_grad = second_order_grad.detach()  # .cpu()

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

    @staticmethod
    @jx.jaxtyped(typechecker=beartype)
    def _get_self_interaction_extra_term(
        func: Callable[
            [jx.Float[Tensor, " input_shape_flattened "]],
            jx.Float[Tensor, ""],
        ],
        inputs_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
        baselines_flattened: jx.Float[Tensor, "batch_size input_shape_flattened "],
        approximation_steps: int,
        batch_size: Optional[int],
        riemann_flavor: riemann_flavors,
        optimize_for_duplicate_interpolation_values: bool,
    ) -> jx.Float[Tensor, "batch_size input_shape_flattened"]:
        """
        if i == j, then

        IH = (xi - xi') sum_l=1^k(sum_p=1^m( df(x' + (l/k) (x - x'))/dxi 1/k/m )) + (xi - xi') (xj - xj') sum_l=1^k(sum_p=1^m( l/k*p/m * df(x' + (l/k) (x - x'))/(dxi dxj) 1/k/m ))

        The second term is already calculated at _get_common_term(), here, we calculate  the first term.
        """
        batch_shape = inputs_flattened.shape[0]
        flattened_input_shape = inputs_flattened.shape[1]

        # this carries the riemann sum
        self_interaction_term = torch.zeros(
            batch_shape, flattened_input_shape, device=inputs_flattened.device
        )

        get_jacobian = torch.vmap(torch.func.jacrev(func=func), chunk_size=batch_size)

        for alphabeta, weight in zip(
            *_get_riemann_interpolation_coefficients(
                approximation_steps=approximation_steps,
                verbose=False,
                riemann_flavor=riemann_flavor,
                optimize_for_duplicate_interpolation_values=optimize_for_duplicate_interpolation_values,
            )
        ):
            # This is one point on the path integral
            interpolation: jx.Float[Tensor, "batch_size input_shape_flattened "] = (
                baselines_flattened
                + alphabeta * (inputs_flattened - baselines_flattened)
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
def _get_riemann_interpolation_coefficients(
    riemann_flavor: riemann_flavors,
    optimize_for_duplicate_interpolation_values: bool,
    approximation_steps: int,
    verbose: bool,
):
    k = approximation_steps
    m = approximation_steps
    alphabetas: list[float] = []
    weights: list[float] = []

    match riemann_flavor:
        case "midpoint_rule":
            for l in range(1, k + 1):  # noqa: E741
                beta: float = (l - 0.5) / k
                for p in range(1, m + 1):
                    alpha: float = (p - 0.5) / m
                    alphabetas.append(beta * alpha)
                    weights.append(1)
        case "trapezoid_rule":
            for l in range(0, k + 1):
                beta: float = l / k
                w_beta = 0.5 if (l == 0 or l == k) else 1.0
                for p in range(0, m + 1):
                    alpha: float = p / m
                    w_alpha = 0.5 if (p == 0 or p == m) else 1.0
                    alphabetas.append(beta * alpha)
                    weights.append(w_beta * w_alpha)  # no /k/m here

        case _:
            raise ValueError(f"Invalid riemann_flavor: {riemann_flavor!r}")

    if optimize_for_duplicate_interpolation_values:
        # Use integer keys to avoid float equality issues.
        # For midpoint: key = (2l-1) * (2p-1), which is exact and unique per unique alphabeta.
        # For trapezoid: key = l * m + p * k (scaled grid indices), same idea.
        # Instead, we bucket by rounding to a safe precision and summing weights.
        merged: dict[int, float] = defaultdict(float)
        # Represent alphabeta as a fraction scaled to avoid floats as keys.
        # alphabeta = (num_l / k) * (num_p / m) = (num_l * num_p) / (k * m)
        # So the unique integer key is just the numerator product.
        # For midpoint: num_l = 2l-1, num_p = 2p-1 → key = (2l-1)*(2p-1)
        # For trapezoid: num_l = l, num_p = p → key = l*m + p  (use pairing, not product,
        # since l*p collisions exist e.g. 1*4 == 2*2)
        #
        # Simpler and flavor-agnostic: recompute integer indices from alphabeta.
        # Just use a dict keyed on the exact rational numerator.
        # We scale alphabeta by k*m (or 2k*2m for midpoint) to get an integer.

        scale = (2 * k) * (
            2 * m
        )  # works for both: midpoint gives odd*odd, trapezoid even*even
        merged_weights: dict[int, float] = defaultdict(float)
        merged_alphabetas: dict[int, float] = {}

        for ab, w in zip(alphabetas, weights):
            key = round(
                ab * scale
            )  # exact for both flavors since ab is a ratio of small ints
            merged_weights[key] += w
            merged_alphabetas[key] = ab  # same ab for all duplicates, safe to overwrite

        before = len(alphabetas)
        alphabetas = list(merged_alphabetas.values())
        weights = list(merged_weights.values())

        if verbose:
            after = len(alphabetas)
            reduction = (1 - after / before) * 100
            print(
                f"[{riemann_flavor}] deduplication reduced calculations by {reduction:.1f}% "
                f"({before} → {after} unique interpolation points)"
            )

    return alphabetas, weights
