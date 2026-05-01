import jaxtyping as jx
from path_explain import PathExplainerTorch
from torch import Tensor
from typing import Callable

import torch


def path_explain_wrapper(
    inputs: jx.Float[Tensor, "batchsize SEQLEN 4"],
    baselines: jx.Float[Tensor, "batchsize SEQLEN 4"],
    model: Callable,
    approximation_steps: int,
    baseline_fill=0.25,
):
    batch_size = inputs.shape[0]
    seqlen = inputs.shape[1]

    def exp_reshaper(x: Tensor):
        x = x.reshape((batch_size, seqlen, 4))
        x = model(x)
        return x

    exp = PathExplainerTorch(exp_reshaper)

    exp_input = inputs.reshape(batch_size, seqlen * 4)
    exp_baseline = baselines.reshape(batch_size, seqlen * 4)

    exp_input.requires_grad_(True)
    exp_baseline.requires_grad_(True)

    exp_ih = exp.interactions(
        exp_input,
        exp_baseline[
            0
        ],  # unlike our api, where we can take one baseline corresponding to one input, path_explain accepts one baseline corresponding to all inputs, hence the subsetting[0] here
        num_samples=approximation_steps,
        use_expectation=False,
    )

    exp_ih_delta = (
        model(inputs) - model(torch.full_like(inputs, baseline_fill)) - exp_ih.sum()
    )

    exp_ih_reshaped = exp_ih.reshape(batch_size, seqlen, 4, seqlen, 4)
    exp_ih_reshaped.shape

    return exp_ih_reshaped, exp_ih_delta
