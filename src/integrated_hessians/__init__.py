from contextlib import AbstractContextManager, nullcontext
from typing import Tuple, Union
from beartype import beartype
from torch import Tensor
import torch
import jaxtyping as jx
from .algorithm.relu_replacer import replace_relu_with_softplus
from .algorithm.just_hessian import get_hessian as get_hessian
from .algorithm.strategies.riemann import RiemannIH as RiemannIH
from .algorithm.strategies.gauss_legendre import GaussQuadratureIH as GaussQuadratureIH
from ._core import (
    PathIntegralStrategy,
    forward_func_type,
    input_type,
    calculated_integrated_hessians,
    integrated_hessians_delta,
    scalar_forward_func_type,
)


class IntegratedHessians:
    @jx.jaxtyped(typechecker=beartype)
    def __init__(
        self,
        forward_func: forward_func_type,  # aka model
        path_integral_strategy: PathIntegralStrategy,
        replace_relu: bool | AbstractContextManager[None] = True,
    ):
        self.forward_func = forward_func
        self.strategy = path_integral_strategy

        if isinstance(replace_relu, bool):
            if replace_relu:
                # context managers from generators can only be used once, so we use a factory to re-generate them every time
                self.replace_relu = lambda: (
                    replace_relu_with_softplus()
                )  # just init with defaults
            else:
                self.replace_relu = lambda: nullcontext()  # does nothing
        else:
            assert isinstance(replace_relu, AbstractContextManager), (
                "replace_relu argument must either be a boolean or a context manager"
            )

    @jx.jaxtyped(typechecker=beartype)
    def get_integrated_hessians(
        self,
        inputs: input_type,
        baselines: input_type,
        target: Union[None, int, Tuple[int, ...]],
        approximation_steps=50,
    ) -> Tuple[
        calculated_integrated_hessians,  # interaction attributions
        integrated_hessians_delta,  # deltas
    ]:
        # setup and assert inputs

        assert len(inputs.shape) >= 2, (
            "inputs must have at least two dimensions, one bath dimension and one or more input dimensions"
        )
        batch_shape = inputs.shape[0]
        input_shape = inputs.shape[1:]
        assert baselines.shape == (batch_shape, *input_shape), (
            "Input tensor and baseline tensor must have the same shape"
        )
        assert inputs.device == baselines.device, (
            "Inputs and the baselines should be on the same device"
        )
        if isinstance(self.forward_func, torch.nn.Module):
            assert inputs.device == next(self.forward_func.parameters()).device, (
                "Model and the inputs should be on the same device"
            )
        #   Reset the autograd history on the tensors
        #   I am not sure if this is necessary, but I am doing it just in case
        inputs = inputs.detach().requires_grad_(True)
        baselines = baselines.detach().requires_grad_(True)

        # convert f to scalar f
        if target is not None:
            scalar_forward_function = self.convert_f_to_scalar_f(
                self.forward_func, target=target
            )
        else:
            scalar_forward_function = self.forward_func

        # call algorithm
        #   only replaces relu if specified, otherwise a null stand-in that does nothing is used
        with self.replace_relu():
            return self.strategy.get_integrated_hessians(
                inputs=inputs,
                baselines=baselines,
                approximation_steps=approximation_steps,
                scalar_forward_function=scalar_forward_function,
            )

    @staticmethod
    @jx.jaxtyped(typechecker=beartype)
    def convert_f_to_scalar_f(
        f,
        target: Union[
            int, Tuple[int, ...]
        ],  # tuple of ints could be used to subset if the output shape is multi dimensional in addition to the batch dimension
    ) -> scalar_forward_func_type:
        @jx.jaxtyped(typechecker=beartype)
        def func_with_scalar_output(
            x: input_type,
        ) -> jx.Float[Tensor, " batch_size "]:
            output: Tensor = f(x)
            if target is not None:
                if isinstance(target, int):
                    output = output[:, target]
                else:
                    output = output[:, *target]
            assert len(output.shape) == 1, (
                "expected a scalar output of shape: (batch_size,)"
            )
            return output

        return func_with_scalar_output
