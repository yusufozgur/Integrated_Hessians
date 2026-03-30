import jaxtyping as jx
from integrated_hessians import get_integrated_hessians
from integrated_hessians.simulation import SimulatedSequence
from integrated_hessians.simulation.custom_additive_and_interactive_effects.test_model import (
    get_test_data,
)
from integrated_hessians.simulation.custom_additive_and_interactive_effects.config import (
    INTEGRATED_HESSIANS_SAMPLING_STEPS,
    TEST_DATA,
    OUT_BEST_MODEL,
    OUT_EXTRACTED_self_interactions_and_pair_interactions_sums,
    DEVICE,
)
import torch
from torch import Tensor
import json
from integrated_hessians.simulation.model import CNNMLP
import numpy as np

BATCH_SIZE = 50


def main():
    # setup
    test_data: list[SimulatedSequence] = get_test_data(TEST_DATA)[:3000]

    model = CNNMLP()
    model.load_state_dict(torch.load(OUT_BEST_MODEL))
    model.eval()
    model = model.to(DEVICE)

    one_hots: jx.Float[Tensor, "batch_size seqlen 4"] = torch.tensor(
        np.array([row.one_hot for row in test_data]), dtype=torch.float
    ).to(DEVICE)
    batch_size = one_hots.shape[0]
    seqlen = one_hots.shape[1]
    names1: list[str] = [row.motif_names[0] for row in test_data]
    names2: list[str] = [row.motif_names[1] for row in test_data]
    masks1: jx.Float[Tensor, "batch_shape seqlen"] = torch.tensor(
        [[int(x) for x in list(row.motif_mask_1)] for row in test_data], device=DEVICE
    )
    masks2: jx.Float[Tensor, "batch_shape seqlen"] = torch.tensor(
        [[int(x) for x in list(row.motif_mask_2)] for row in test_data], device=DEVICE
    )
    baselines: jx.Float[Tensor, "batch_size seqlen 4"] = torch.full_like(one_hots, 0.25)

    # input_minus_baseline_preds: jx.Float[Tensor, " batch_shape "] = (
    #     (model(one_hots) - model(baselines)).squeeze(-1).detach()
    # )
    model_preds = model(one_hots).detach().reshape(batch_size)

    # Calculate Integrated Hessians

    integ_hess_interactions: jx.Float[Tensor, "batch_size seqlen 4 seqlen 4"]
    ih_delta: jx.Float[Tensor, " batch_size "]
    integ_hess_interactions, ih_delta = get_integrated_hessians(
        model=model,
        inputs=one_hots,
        baselines=baselines,
        target=0,
        approximation_steps=INTEGRATED_HESSIANS_SAMPLING_STEPS,
        optimize_for_duplicate_interpolation_values=True,
        batch_size=BATCH_SIZE,
    )

    integ_hess_interactions = integ_hess_interactions.detach()
    ih_delta = ih_delta.detach()

    # Important, all positions, including real and unreal, are considered. We do not subset for existing nucleotides, as non-existing nucleotide positions affect the outcome as well.

    ih_mask_pair_1 = (
        integ_hess_interactions
        * masks1.reshape(batch_size, seqlen, 1, 1, 1)
        * masks2.reshape(batch_size, 1, 1, seqlen, 1)
    )
    ih_mask_pair_2 = (
        integ_hess_interactions
        * masks1.reshape(batch_size, 1, 1, seqlen, 1)
        * masks2.reshape(batch_size, seqlen, 1, 1, 1)
    )
    ih_mask_self_interaction_1 = (
        integ_hess_interactions
        * masks1.reshape(batch_size, seqlen, 1, 1, 1)
        * masks1.reshape(batch_size, 1, 1, seqlen, 1)
    )
    ih_mask_self_interaction_2 = (
        integ_hess_interactions
        * masks2.reshape(batch_size, seqlen, 1, 1, 1)
        * masks2.reshape(batch_size, 1, 1, seqlen, 1)
    )

    # sum
    ih_mask_pair_1_sums = ih_mask_pair_1.reshape(batch_size, -1).sum(dim=1)
    ih_mask_pair_2_sums = ih_mask_pair_2.reshape(batch_size, -1).sum(dim=1)
    ih_mask_self_interaction_1_sums = ih_mask_self_interaction_1.reshape(
        batch_size, -1
    ).sum(dim=1)
    ih_mask_self_interaction_2_sums = ih_mask_self_interaction_2.reshape(
        batch_size, -1
    ).sum(dim=1)

    out_list = []
    for (
        test_row,
        sum_pair1,
        sum_pair2,
        sum_selfinteract1,
        sum_selfinteract2,
        pred,
    ) in zip(
        test_data,
        ih_mask_pair_1_sums,
        ih_mask_pair_2_sums,
        ih_mask_self_interaction_1_sums,
        ih_mask_self_interaction_2_sums,
        model_preds,
    ):
        out_list.append(
            {
                "name1": test_row.motif_names[0],
                "name2": test_row.motif_names[1],
                "sum_of_pairs": float(sum_pair1 + sum_pair2),
                "sum_self_interaction_1": float(sum_selfinteract1),
                "sum_self_interaction_2": float(sum_selfinteract2),
                "prediction": float(pred),
                "phenotype": test_row.phenotype,
            }
        )

    with open(OUT_EXTRACTED_self_interactions_and_pair_interactions_sums, "w") as f:
        json.dump(out_list, f, indent=4)

    # also save deltas integrated hessians

    with open(
        OUT_EXTRACTED_self_interactions_and_pair_interactions_sums.with_stem(
            f"{OUT_EXTRACTED_self_interactions_and_pair_interactions_sums.stem}_deltas"
        ).with_suffix(".json"),
        "w",
    ) as f:
        json.dump(ih_delta.tolist(), f, indent=4)


if __name__ == "__main__":
    main()
