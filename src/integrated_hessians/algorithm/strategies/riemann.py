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


class RiemannIH(PathIntegralStrategy):
    interpolation_coefficients_mode: Literal[
        "default", "multiplication_table_optimized"
    ]

    @jx.jaxtyped(typechecker=beartype)
    def __init__(
        self,
        optimize_for_duplicate_interpolation_values: bool = True,
        batch_size: Optional[int] = None,
    ):
        self.batch_size = batch_size

        match optimize_for_duplicate_interpolation_values:
            case True:
                self.interpolation_coefficients_mode = "multiplication_table_optimized"
            case False:
                self.interpolation_coefficients_mode = "default"

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
            interpolation_coefficients_mode=self.interpolation_coefficients_mode,
            batch_size=self.batch_size,
        ).detach()

        # self_interaction_extra_terms are calculated for diagonals(i==j), it exists due to the chain rule of calculus when you take d(IG(xi))/dxi.
        self_interaction_extra_term: jx.Float[
            Tensor, "batch_size input_shape_flattened"
        ] = self._get_self_interaction_extra_term(
            func=func_vmap_compatible_transient_batch_dim,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            approximation_steps=approximation_steps,
            interpolation_coefficients_mode=self.interpolation_coefficients_mode,
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
        interpolation_coefficients_mode: Literal[
            "default", "multiplication_table_optimized"
        ],
        batch_size: Optional[int],
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
            mode=interpolation_coefficients_mode,
            approximation_steps=approximation_steps,
            verbose=True,
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
        interpolation_coefficients_mode: Literal[
            "default", "multiplication_table_optimized"
        ],
        batch_size: Optional[int],
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
                mode=interpolation_coefficients_mode,
                approximation_steps=approximation_steps,
                verbose=False,
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
