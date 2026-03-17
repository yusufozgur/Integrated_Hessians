from contextlib import contextmanager
from typing import Callable, Annotated, Literal
from beartype import beartype
from torch import Tensor
import torch
import torch.nn.functional as F
import jaxtyping as jx


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
        hessian_result: Annotated[
            Tensor, "batch_size *input_shape batch_size *input_shape"
        ] = torch.autograd.functional.hessian(forward_func, input, strict=True)

    return hessian_result
