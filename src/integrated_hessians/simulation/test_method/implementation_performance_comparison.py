"""
Across the datapoints in the test set, compare the performance of different methods.

Methods:
-   Path explain
-   Integrated_Hessians(us), with naive riemann
-   Integrated_Hessians(us), with cached riemann
-   Integrated_Hessians(us), with quadrature

Measure
-   We do not optimize for finding which sampling sizes the implementations reach a delta threshold, as the sampling size vs delta curve is not consistent across samples, leading to many failed root finding runs. Instead, we just cap at a certain constant sampling size where the deltas did not saturate yet, so there would be both differences in deltas across method and also differences in compute time across methods
-   measure 1: delta per constant sampling size
-   measure 2: compute time needed for that sampling size, this could also be a curve where x is diff sampling sizes and y is compute times. It would show the dynamics of the compute time.

Dataset: Custom sim with expanded distribution
"""

import json
import functools
from os import cpu_count
from pathlib import Path
from typing import Callable
import jaxtyping as jx
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from integrated_hessians import get_integrated_hessians
from integrated_hessians.simulation import SimulatedSequence
from integrated_hessians.simulation.model import CNNMLP
from integrated_hessians.simulation.test_method.path_explain_wrapper import (
    path_explain_wrapper,
)
from integrated_hessians.simulation.test_model import get_test_data
from dataclasses import dataclass, asdict

from integrated_hessians.simulation.train_model import MotifInteractionsDataset

# Configuration
CONFIGPATH = (
    "src/integrated_hessians/simulation/configs/custom_expanded_distribution.json"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # keep batch size at 1 as otherwise the deltas of the janizek method just are bad, probably due to bad seperation of samples inside the implementation
NUM_OF_ROWS = 10
BASELINE_FILL = 0.25
SAVE_test_deltas_per_impl = (
    "src/integrated_hessians/simulation/test_method/test_deltas_per_impl.json"
)


implementations = {
    "ih_naive_riemann": {
        "f": functools.partial(
            get_integrated_hessians,
            target=0,
            optimize_for_duplicate_interpolation_values=False,
        ),
        "approx_steps": 20,
    },
    "ih_cached_riemann": {
        "f": functools.partial(
            get_integrated_hessians,
            target=0,
            optimize_for_duplicate_interpolation_values=True,
        ),
        "approx_steps": 20,
    },
    "janizeketal": {
        "f": functools.partial(
            path_explain_wrapper,
        ),
        "approx_steps": 20**2,
    },
}


@dataclass
class PerformanceTest:
    delta: float
    function_calls: int


def main():

    print(f"Using device: {DEVICE}")

    # Load model and data
    with open(CONFIGPATH, "r") as f:
        config = json.load(f)

    model = CNNMLP(
        sequence_length=config["SEQLEN"],
        width_multiplier=config["MODEL_WIDTH_MULTIPLIER"],
    )
    model.load_state_dict(torch.load(config["OUT_BEST_MODEL"], weights_only=True))
    model.eval()
    model = model.to(DEVICE)

    test_data = get_test_data(Path(config["TEST_DATA"]), config["SEQLEN"])
    model = model.to(DEVICE)

    print(f"Test data size: {len(test_data)}")
    print(f"Config: {config['NAME']}")

    # Run performance comparisons

    deltas = {
        impl_name: [
            asdict(x)
            for x in get_delta_per_test_row(
                test_data=test_data,
                implementation_config=impl_config,
                model=model,
                config=config,
            )
        ]
        for impl_name, impl_config in implementations.items()
    }

    with open(SAVE_test_deltas_per_impl, "w") as f:
        json.dump(deltas, f, indent=4)


def get_delta_per_test_row(
    implementation_config: dict,
    test_data: list[SimulatedSequence],
    model: Callable,
    config: dict,
) -> list[PerformanceTest]:
    num_of_function_calls = 0

    def model_call_counter(*args, **kwargs):
        nonlocal num_of_function_calls
        num_of_function_calls += 1
        return model(*args, **kwargs)

    dataset = MotifInteractionsDataset(
        input=Path(config["TEST_DATA"]),
        SEQLEN=config["SEQLEN"],
        EXPAND_DATA_DISTRIBUTION_ALONG_BASELINE_TO_INPUT_PATH=False,
    )

    dataset = Subset(dataset, range(NUM_OF_ROWS))

    num_of_cores = cpu_count()
    if num_of_cores is None:
        num_of_cores = 1
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, num_workers=num_of_cores
    )
    results = []
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(DEVICE).to(torch.float)
        baselines = torch.full_like(inputs, fill_value=BASELINE_FILL).to(DEVICE)

        ih_delta: Tensor
        integ_hess_result, ih_delta = implementation_config["f"](
            approximation_steps=implementation_config["approx_steps"],
            inputs=inputs,
            baselines=baselines,
            model=model_call_counter,
        )
        ih_deltas = ih_delta.detach().tolist()  # [0] to remove the batch dim

        results += [
            PerformanceTest(delta=x, function_calls=num_of_function_calls)
            for x in ih_deltas
        ]
        num_of_function_calls = 0
    return results


if __name__ == "__main__":
    main()
