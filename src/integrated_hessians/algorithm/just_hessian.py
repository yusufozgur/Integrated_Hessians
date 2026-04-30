from typing import Callable, Annotated
from beartype import beartype
from torch import Tensor
import torch
import jaxtyping as jx
from integrated_hessians.algorithm.relu_replacer import replace_relu_with_softplus


@jx.jaxtyped(typechecker=beartype)
def get_hessian(
    model: Callable,
    input: jx.Float[Tensor, " batch_size *input_shape"],
    target: int,
) -> Annotated[Tensor, "batch_size *input_shape batch_size *input_shape"]:
    """
    Utility function for the notebook.
    """

    def forward_func(x):
        return model(x)[target]

    with replace_relu_with_softplus():
        hessian_result: jx.Float[
            Tensor, "batch_size *input_shape batch_size *input_shape"
        ] = torch.autograd.functional.hessian(forward_func, input, strict=True)  # type: ignore

    # jaxtyping does not support multiple variadic specifiers (*name,). So, instead, I am checking the output is correct shape via an assert statement
    assert hessian_result.shape == (*input.shape, *input.shape)

    return hessian_result
