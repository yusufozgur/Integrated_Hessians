"""
GaussQuadratureIH: Gauss-Legendre quadrature strategy for Integrated Hessians.

Instead of approximating the double path integral with a uniform Riemann grid,
this class uses Gauss-Legendre quadrature nodes and weights on [0, 1] × [0, 1].
For smooth integrands this converges much faster than Riemann sums —
O(n^{2k}) error for k-point GL vs O(n^{-2}) for midpoint Riemann —
meaning you typically need far fewer `approximation_steps` to reach the same accuracy.

Drop-in replacement for RiemannIH:

    strategy = GaussQuadratureIH(n_points=10, batch_size=32)
    attributions, delta = calculated_integrated_hessians(
        model, inputs, baselines, strategy, approximation_steps=10
    )
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import jaxtyping as jx
from beartype import beartype
from torch import Size, Tensor
from tqdm import tqdm

from integrated_hessians._core import (
    PathIntegralStrategy,
    calculated_integrated_hessians,
    integrated_hessians_delta,
    scalar_forward_func_type,
    input_type,
)

# ---------------------------------------------------------------------------
# Type aliases (mirrors the ones in the Riemann module)
# ---------------------------------------------------------------------------

flattened_input_type = jx.Float[Tensor, "batch_size input_shape_flattened"]
flattened_hessian_type = jx.Float[
    Tensor, "batch_size input_shape_flattened input_shape_flattened"
]
flattened_forward_func_type = Callable[
    [flattened_input_type],
    jx.Float[Tensor, " batch_size"],
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GaussQuadratureIH(PathIntegralStrategy):
    """
    Integrated-Hessian strategy using product Gauss-Legendre quadrature.

    The double path integral
        ∫₀¹ ∫₀¹  αβ · H(x' + αβ(x − x')) dα dβ
    (and its first-order counterpart for diagonal self-interaction terms) is
    approximated with an `approximation_steps`-point Gauss-Legendre rule in
    each dimension, giving a grid of `approximation_steps²` quadrature nodes.

    Parameters
    ----------
    batch_size : int | None
        Maximum number of samples to pass to the Hessian/Jacobian vmapped
        function at once.  Set this if GPU memory is tight.  ``None`` means
        process all samples in one shot.

    Notes
    -----
    * There are no duplicate interpolation values with GL nodes, so no
      deduplication step is needed (unlike the Riemann trapezoid rule).
    * The `approximation_steps` argument of ``get_integrated_hessians`` is
      reinterpreted as the *number of GL points per dimension*.  For the same
      wall-clock budget, a GL run with n points is far more accurate than a
      Riemann run with n steps, because GL achieves exact integration for
      polynomials of degree ≤ 2n − 1.
    """

    @jx.jaxtyped(typechecker=beartype)
    def __init__(self, batch_size: Optional[int] = None):
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

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
        """
        Parameters
        ----------
        scalar_forward_function :
            Function mapping (batch_size, *input_shape) → (batch_size,).
        inputs, baselines :
            Tensors of shape (batch_size, *input_shape).
        approximation_steps :
            Number of Gauss-Legendre nodes **per dimension** of the double
            integral.  Total quadrature evaluations = approximation_steps².
            Values in the range 5–20 are typically sufficient for smooth models.

        Returns
        -------
        interaction_attributions :
            Shape (batch_size, *input_shape, *input_shape).
        delta :
            Completeness gap: f(input) − f(baseline) − Σ attributions.
            Smaller is better; near-zero confirms the approximation is accurate.
        """
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

        # vmap-compatible wrapper (strips the batch dim for torch.func.hessian)
        @jx.jaxtyped(typechecker=beartype)
        def func_vmap_compatible(
            x: jx.Float[Tensor, "input_shape_flattened"],
        ) -> jx.Float[Tensor, ""]:
            assert x.shape == (input_shape_flattened,)
            scalar_output = flattened_scalar_forward_func(x.unsqueeze(0))
            return scalar_output.squeeze(0)

        # ---- common term: ∫∫ αβ H(x'+αβ(x−x')) dα dβ --------------------
        common_term: flattened_hessian_type = self._get_common_term(
            func=func_vmap_compatible,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            n_points=approximation_steps,
            batch_size=self.batch_size,
        ).detach()

        # ---- self-interaction extra term (diagonal only) ------------------
        self_interaction_extra_term: jx.Float[
            Tensor, "batch_size input_shape_flattened"
        ] = self._get_self_interaction_extra_term(
            func=func_vmap_compatible,
            inputs_flattened=inputs_flattened,
            baselines_flattened=baselines_flattened,
            n_points=approximation_steps,
            batch_size=self.batch_size,
        ).detach()

        # Combine: off-diagonal from common_term, diagonal += self_interaction
        interaction_attributions: flattened_hessian_type = (
            common_term + self_interaction_extra_term.diag_embed()
        )

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    @jx.jaxtyped(typechecker=beartype)
    def _get_function_with_flat_to_original_shape_conversion(
        f: scalar_forward_func_type,
        input_shape: Size,
        input_shape_flattened: int,
    ) -> flattened_forward_func_type:
        @jx.jaxtyped(typechecker=beartype)
        def flattened_forward_func(
            x: flattened_input_type,
        ) -> jx.Float[Tensor, " batch_size "]:
            assert len(x.shape) == 2
            assert x.shape[1] == input_shape_flattened
            x_batch_size = x.shape[0]
            output = f(x.reshape(x_batch_size, *input_shape))
            assert output.shape == (x_batch_size,), (
                "Could not reduce output dimension to scalars; "
                "the target used for subsetting may be incorrect."
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
        f_input = func(inputs_flattened)
        f_baseline = func(baselines_flattened)
        out_diff = f_input - f_baseline
        attr_sum = interaction_attributions.sum(dim=-1).sum(dim=-1)
        return out_diff - attr_sum

    @staticmethod
    @jx.jaxtyped(typechecker=beartype)
    def _get_common_term(
        func: Callable[
            [jx.Float[Tensor, " input_shape_flattened "]],
            jx.Float[Tensor, ""],
        ],
        inputs_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
        baselines_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
        n_points: int,
        batch_size: Optional[int],
    ) -> jx.Float[Tensor, "batch_size input_shape_flattened input_shape_flattened"]:
        """
        Approximates, for all (i, j) pairs (including i == j):

            (xi − xi')(xj − xj') ∫₀¹ ∫₀¹ αβ · ∂²f/∂xi∂xj(x' + αβ(x−x')) dα dβ

        using an n_points × n_points Gauss-Legendre product rule.

        This is the *common* term shared by both off-diagonal (i ≠ j) and
        diagonal (i == j) entries.  The self-interaction extra term (calculated
        separately) is added only to the diagonal afterward.
        """
        batch_shape = inputs_flattened.shape[0]
        flat_dim = inputs_flattened.shape[1]

        second_order_sensitivity = torch.zeros(
            batch_shape,
            flat_dim,
            flat_dim,
            device=inputs_flattened.device,
        )

        get_second_order_grad = torch.vmap(torch.func.hessian(func=func))

        alphabetas, gl_weights = _get_gauss_interpolation_coefficients(
            n_points=n_points, verbose=True
        )

        for alphabeta, gl_weight in tqdm(
            zip(alphabetas, gl_weights),
            total=len(alphabetas),
            desc="GaussQuadratureIH common term",
        ):
            interpolation: jx.Float[Tensor, "batch_size input_shape_flattened"] = (
                baselines_flattened
                + alphabeta * (inputs_flattened - baselines_flattened)
            )

            if batch_size is None:
                second_order_grad: jx.Float[
                    Tensor, "batch_size input_shape_flattened input_shape_flattened"
                ] = get_second_order_grad(interpolation)
            else:
                chunks = []
                for chunk in interpolation.split(batch_size):
                    chunks.append(get_second_order_grad(chunk).detach())
                second_order_grad = torch.cat(chunks)

            second_order_grad = second_order_grad.detach()

            # Gauss weight already encodes the ∫dα dβ measure;
            # alphabeta is the αβ factor in the integrand itself.
            second_order_sensitivity += second_order_grad * alphabeta * gl_weight

        diff = inputs_flattened - baselines_flattened
        outer_product = diff.unsqueeze(dim=1) * diff.unsqueeze(dim=2)
        return second_order_sensitivity * outer_product

    @staticmethod
    @jx.jaxtyped(typechecker=beartype)
    def _get_self_interaction_extra_term(
        func: Callable[
            [jx.Float[Tensor, " input_shape_flattened "]],
            jx.Float[Tensor, ""],
        ],
        inputs_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
        baselines_flattened: jx.Float[Tensor, "batch_size input_shape_flattened"],
        n_points: int,
        batch_size: Optional[int],
    ) -> jx.Float[Tensor, "batch_size input_shape_flattened"]:
        """
        Approximates the extra diagonal term arising from the chain rule when
        differentiating IG(xi) with respect to xi:

            (xi − xi') ∫₀¹ ∫₀¹ ∂f/∂xi(x' + αβ(x−x')) dα dβ

        using an n_points × n_points Gauss-Legendre product rule.
        """
        batch_shape = inputs_flattened.shape[0]
        flat_dim = inputs_flattened.shape[1]

        self_interaction_term = torch.zeros(
            batch_shape,
            flat_dim,
            device=inputs_flattened.device,
        )

        get_jacobian = torch.vmap(torch.func.jacrev(func=func), chunk_size=batch_size)

        alphabetas, gl_weights = _get_gauss_interpolation_coefficients(
            n_points=n_points, verbose=False
        )

        for alphabeta, gl_weight in zip(alphabetas, gl_weights):
            interpolation: jx.Float[Tensor, "batch_size input_shape_flattened"] = (
                baselines_flattened
                + alphabeta * (inputs_flattened - baselines_flattened)
            )

            first_order_grad: jx.Float[Tensor, "batch_size input_shape_flattened"] = (
                get_jacobian(interpolation)
            )

            first_order_grad = first_order_grad.detach()

            # Gauss weight encodes the full ∫dα dβ measure over [0,1]².
            self_interaction_term += first_order_grad * gl_weight

        diff = inputs_flattened - baselines_flattened
        return diff * self_interaction_term


# ---------------------------------------------------------------------------
# Quadrature coefficient helper
# ---------------------------------------------------------------------------


@jx.jaxtyped(typechecker=beartype)
def _get_gauss_interpolation_coefficients(
    n_points: int,
    verbose: bool,
) -> tuple[list[float], list[float]]:
    """
    Build the (alphabeta, weight) pairs for an n_points × n_points product
    Gauss-Legendre quadrature rule on [0, 1] × [0, 1].

    For each pair of nodes (α_i, β_j):
      * ``alphabeta`` = α_i · β_j   — the combined interpolation coefficient
                                       feeding into x' + αβ(x − x')
      * ``weight``    = w_i · w_j   — the product quadrature weight,
                                       already scaled to integrate over [0,1]²

    The GL nodes/weights on [−1, 1] are transformed to [0, 1] by
        node₀₁  = (node₋₁₁ + 1) / 2
        weight₀₁ = weight₋₁₁  / 2

    Parameters
    ----------
    n_points : int
        Number of GL nodes per dimension.  Total quadrature points = n_points².
        GL with n points integrates polynomials of degree ≤ 2n − 1 exactly.
    verbose : bool
        If True, print the total number of quadrature points.

    Returns
    -------
    alphabetas : list[float]
        Interpolation coefficients, one per quadrature point.
    weights : list[float]
        Corresponding quadrature weights.
    """
    # GL nodes and weights on [-1, 1]
    nodes_m1p1, weights_m1p1 = np.polynomial.legendre.leggauss(n_points)

    # Transform to [0, 1]
    nodes_01: np.ndarray = (nodes_m1p1 + 1.0) / 2.0  # shape (n_points,)
    weights_01: np.ndarray = weights_m1p1 / 2.0  # shape (n_points,)

    # Build product rule over (alpha, beta) ∈ [0,1] × [0,1]
    alphabetas: list[float] = []
    weights: list[float] = []

    for beta, w_beta in zip(nodes_01.tolist(), weights_01.tolist()):
        for alpha, w_alpha in zip(nodes_01.tolist(), weights_01.tolist()):
            alphabetas.append(float(beta * alpha))
            weights.append(float(w_beta * w_alpha))

    if verbose:
        print(
            f"[GaussQuadratureIH] {n_points}×{n_points} product GL rule → "
            f"{len(alphabetas)} quadrature points "
            f"(exact for polynomials of degree ≤ {2 * n_points - 1})"
        )

    return alphabetas, weights
