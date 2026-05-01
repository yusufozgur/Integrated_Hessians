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
from pathlib import Path
from typing import Callable
import jaxtyping as jx
import torch
from torch import Tensor
from tqdm import tqdm
from integrated_hessians import get_integrated_hessians
from integrated_hessians.simulation import SimulatedSequence
from integrated_hessians.simulation.model import CNNMLP
from integrated_hessians.simulation.test_method.path_explain_wrapper import (
    path_explain_wrapper,
)
from integrated_hessians.simulation.test_model import get_test_data
from dataclasses import dataclass, asdict

# Configuration
CONFIGPATH = (
    "src/integrated_hessians/simulation/configs/custom_expanded_distribution.json"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_OF_ROWS = 5
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
                test_data=test_data, implementation_config=impl_config, model=model
            )
        ]
        for impl_name, impl_config in implementations.items()
    }

    with open(SAVE_test_deltas_per_impl, "w") as f:
        json.dump(deltas, f, indent=4)


def get_delta_per_test_row(
    implementation_config: dict, test_data: list[SimulatedSequence], model: Callable
) -> list[PerformanceTest]:
    num_of_function_calls = 0

    def model_call_counter(*args, **kwargs):
        nonlocal num_of_function_calls
        num_of_function_calls += 1
        return model(*args, **kwargs)

    results = []
    for test_data_idx in tqdm(range(NUM_OF_ROWS)):
        test_row: SimulatedSequence = test_data[test_data_idx]
        input: jx.Float[Tensor, "1 alphabet_length sequence_length"] = (
            torch.from_numpy(test_row.one_hot).unsqueeze(0).type(torch.float).to(DEVICE)
        )
        baseline = torch.full_like(input, fill_value=BASELINE_FILL).to(DEVICE)

        integ_hess_result, ih_delta = implementation_config["f"](
            approximation_steps=implementation_config["approx_steps"],
            inputs=input,
            baselines=baseline,
            model=model_call_counter,
        )
        ih_delta = float(ih_delta[0])  # [0] to remove the batch dim

        results.append(
            PerformanceTest(delta=ih_delta, function_calls=num_of_function_calls)
        )
    return results


if __name__ == "__main__":
    main()
