"""
Across the datapoints in the test set, compare the performance of different methods.

Methods:
-   Path explain
-   Integrated_Hessians(us), with naive riemann
-   Integrated_Hessians(us), with cached riemann
-   Integrated_Hessians(us), with quadrature

Measure
-   Number of path samples needed to reach delta of 0.01
-   Compute time, for this, use the num of samples needed to reach delta 0.01 from the previous step

Dataset: Custom sim with expanded distribution
"""

from doctest import UnexpectedException
import json
import functools
from scipy.optimize import root_scalar, RootResults
from pathlib import Path
from typing import Any, Callable
import jaxtyping as jx
import torch
from torch import Tensor
from tqdm import tqdm
from integrated_hessians import get_integrated_hessians
from integrated_hessians.simulation import SimulatedSequence
from integrated_hessians.simulation.model import CNNMLP
from integrated_hessians.simulation.test_model import get_test_data
from path_explain import PathExplainerTorch


# Configuration
CONFIGPATH = (
    "src/integrated_hessians/simulation/configs/custom_expanded_distribution.json"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_DELTA = 0.01
TARGET_DELTA_TOLERANCE = 0.001
NUM_OF_ROWS = 5
BASELINE_FILL = 0.25
MIN_SAMPLING_SIZE = 1
MAX_SAMPLING_SIZE = 50
MAXITER = 10
SAVE_sample_sizes_where_target_delta_is_reached = "src/integrated_hessians/simulation/test_method/sample_sizes_where_target_delta_is_reached.json"


def main():
    """Main performance comparison workflow."""

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

    # prepare implementations
    implementations = {
        "ih_naive_riemann": functools.partial(
            get_integrated_hessians,
            model=model,
            target=0,
            optimize_for_duplicate_interpolation_values=False,
        ),
        "ih_cached_riemann": functools.partial(
            get_integrated_hessians,
            model=model,
            target=0,
            optimize_for_duplicate_interpolation_values=True,
        ),
    }

    # Run performance comparisons

    per_impl_sample_sizes_where_target_delta_is_reached = (
        get_sample_sizes_where_target_delta_is_reached(
            test_data=test_data, implementations=implementations
        )
    )


def get_sample_sizes_where_target_delta_is_reached(
    test_data, implementations: dict[str, Any]
):

    sample_sizes_where_target_delta_is_reached = {
        impl_name: loop_over_test_datapoints(
            impl_f,
            test_data=test_data,
        )
        for impl_name, impl_f in implementations.items()
    }

    sample_sizes_where_target_delta_is_reached = {
        k: [
            {
                "idx": idx,
                "converged": getattr(x, "converged"),
                "flag": getattr(x, "flag"),
                "function_calls": getattr(x, "function_calls"),
                "iterations": getattr(x, "iterations"),
                "root": round(getattr(x, "root")),
            }
            if isinstance(x, RootResults)
            else f"{type(x)}:{str(x)}"
            for idx, x in enumerate(list_of_root_results_per_testted_implementation)
        ]
        for k, list_of_root_results_per_testted_implementation in sample_sizes_where_target_delta_is_reached.items()
    }

    with open(SAVE_sample_sizes_where_target_delta_is_reached, "w") as f:
        json.dump(sample_sizes_where_target_delta_is_reached, f, indent=4)

    return sample_sizes_where_target_delta_is_reached


def loop_over_test_datapoints(implementation: Callable, test_data) -> list[RootResults]:
    results = []
    for test_data_idx in tqdm(range(NUM_OF_ROWS)):
        test_row: SimulatedSequence = test_data[test_data_idx]
        print(test_row)
        input: jx.Float[Tensor, "1 alphabet_length sequence_length"] = (
            torch.from_numpy(test_row.one_hot).unsqueeze(0).type(torch.float).to(DEVICE)
        )
        baseline = torch.full_like(input, fill_value=BASELINE_FILL)

        # root finding algos do not work with integers, they work with floats.
        # so, we round the x to integer
        # but then, they may try multiple x values that will result in the same int
        # this cache remove duplicated operation because of that
        cache = dict()

        def implementation_w_samples_set(approximation_steps):
            print(f"{approximation_steps=:.3f}")
            approximation_steps = round(approximation_steps)
            if approximation_steps in cache:
                return cache[approximation_steps]
            integ_hess_result, ih_delta = implementation(
                approximation_steps=approximation_steps,
                inputs=input,
                baselines=baseline,
            )
            ih_delta = float(ih_delta[0])  # [0] to remove the batch dim
            distance_from_desired_delta = ih_delta - TARGET_DELTA
            print(f"delta: {ih_delta: .3f}")
            cache[approximation_steps] = distance_from_desired_delta
            return distance_from_desired_delta

        result = find_convergence_point(
            implementation_w_samples_set,
        )
        results.append(result)
    return results


def find_convergence_point(
    f,
):
    """
    Given a function, we use root finding algorithms to find for which sampling size the function f returns a target scalar of desired value, which will be the delta from the path integral method. For example, if target_delta == 0.01, we will find at which sampling size it becomes that value.
    """

    min_sampling_size = float(MIN_SAMPLING_SIZE)
    max_sampling_size = float(MAX_SAMPLING_SIZE)

    # the bracket based root finding algos require left and right bracket ends to lead to function returning different signs
    # that is not satisfied in some datapoint that I saw, so here, if that is the case, I am increasing sampling size until the sign is positive
    if f(min_sampling_size) < 0:
        while f(min_sampling_size) < 0:
            min_sampling_size += 1
            if min_sampling_size >= MIN_SAMPLING_SIZE + 15:
                return Exception(
                    f"left bracket is increased 15 times but sign is still negative, {min_sampling_size=} and {f(min_sampling_size)=}"
                )

    # Root finding algorithms first calculate f(left_bracket) and f(right_bracket)
    # This means that if right bracket is too big, it takes a long time
    # However, I expect the delta to converge in relatively low sample sizes
    # So we will first try lower sample sizes, and if it doesnt converge, increase the sample size
    for upper_limit in (
        # max_sampling_size / 8,
        max_sampling_size / 4,
        max_sampling_size / 2,
        max_sampling_size,
    ):
        try:
            optim_res: RootResults = root_scalar(
                f=f,
                bracket=[min_sampling_size, upper_limit],
                maxiter=MAXITER,
                xtol=TARGET_DELTA_TOLERANCE,
                method="toms748",
                options={"disp": True},
            )
            if optim_res.converged:
                break
        except ValueError as e:
            if upper_limit is not max_sampling_size:
                # A value error is reached, this could be ''ValueError: f(a) and f(b) must have different signs'' error. We will continue with a bigger value for the right side of the bracket
                continue
            else:
                return e
        pass
    print(optim_res)  # pyright: ignore[reportPossiblyUnboundVariable]
    return optim_res  # pyright: ignore[reportPossiblyUnboundVariable]


if __name__ == "__main__":
    main()
