from typing import Callable, Annotated, Tuple
from torch import Tensor
import jaxtyping as jx
from abc import ABC, abstractmethod


input_type = jx.Float[Tensor, " batch_size *input_shape "]
forward_func_type = Callable[
    [input_type],
    jx.Float[Tensor, " batch_size *output_shape "],
]
scalar_forward_func_type = Callable[
    [input_type],
    jx.Float[Tensor, " batch_size "],
]
calculated_integrated_hessians = Annotated[Tensor, "batch_size *input_shape"]
integrated_hessians_delta = Annotated[Tensor, "batch_size"]


class PathIntegralStrategy(ABC):
    """
    Different methods to approximate the path integral are available, such as
        -   naive riemann
        -   cached riemann
        -   gauss legendre quadrature
    Implementation of those methods must inherit this ABC.
    """

    @abstractmethod
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
        pass
