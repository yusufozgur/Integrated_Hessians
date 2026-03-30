from collections import defaultdict

import jaxtyping as jx
from beartype import beartype
from integrated_hessians import get_integrated_hessians
from integrated_hessians.simulation import SimulatedSequence
from integrated_hessians.simulation.custom_additive_and_interactive_effects.test_model import (
    get_test_data,
)
from integrated_hessians.simulation.custom_additive_and_interactive_effects.config import (
    INTEGRATED_HESSIANS_SAMPLING_STEPS,
    TEST_DATA,
    OUT_BEST_MODEL,
    OUT_EXTRACTED_ADDITIVE_EFFECTS,
    OUT_EXTRACTED_INTERACTIVE_EFFECTS,
    DEVICE,
)
import torch
from torch import Tensor
from captum.attr import IntegratedGradients
import json
from integrated_hessians.simulation.model import CNNMLP
import numpy as np

BATCH_SIZE = 50


def main():
    test_data: list[SimulatedSequence] = get_test_data(TEST_DATA)[:1000]

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

    ig = IntegratedGradients(model, multiply_by_inputs=True)
    ig_attributions: jx.Float[Tensor, "batch_size seqlen 4"]
    ig_delta: jx.Float[Tensor, " batch_size "]
    ig_attributions, ig_delta = ig.attribute(
        one_hots,
        baselines,
        return_convergence_delta=True,
        internal_batch_size=20000,
    )

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

    # zero out unreal positions
    ig_attributions = ig_attributions * one_hots
    integ_hess_interactions = (
        one_hots.reshape(batch_size, seqlen, 4, 1, 1)
        * integ_hess_interactions
        * one_hots.reshape(batch_size, 1, 1, seqlen, 4)
    )

    # zero out places outside the masks
    ig_attributions_masks1 = ig_attributions * masks1.reshape(batch_size, seqlen, 1)
    ig_attributions_masks2 = ig_attributions * masks2.reshape(batch_size, seqlen, 1)

    #   for interaction values, it is symmetric so it shouldnt matter which order we multiply with the motif mask

    ih_interactions_masked = (
        integ_hess_interactions
        * masks1.reshape(batch_size, seqlen, 1, 1, 1)
        * masks2.reshape(batch_size, 1, 1, seqlen, 1)
    )

    # sum

    additive_effects_mask1 = ig_attributions_masks1.sum(dim=[1, 2])
    additive_effects_mask2 = ig_attributions_masks2.sum(dim=[1, 2])

    interactive_effects_masked = ih_interactions_masked.sum(dim=[1, 2, 3, 4])

    # normalize based on f-f', which is equal to sum of ig/ih according to completeness axioms
    # additive_effects_mask1 = additive_effects_mask1 / input_minus_baseline_preds
    # additive_effects_mask2 = additive_effects_mask2 / input_minus_baseline_preds
    # interactive_effects_masked = interactive_effects_masked / input_minus_baseline_preds
    #

    motif_attribution_sums = defaultdict(float)
    motif_pairs_interactions_sums = defaultdict(float)

    for name, additive_effect in zip(names1, additive_effects_mask1):
        motif_attribution_sums[name] += float(additive_effect)
    for name, additive_effect in zip(names2, additive_effects_mask2):
        motif_attribution_sums[name] += float(additive_effect)

    for name1, name2, interactive_effect in zip(
        names1, names2, interactive_effects_masked
    ):
        motif_pairs_interactions_sums[frozenset([name1, name2])] += float(
            interactive_effect
        )

    with open(OUT_EXTRACTED_ADDITIVE_EFFECTS, "w") as f:
        # sort
        motif_attribution_sums = dict(
            sorted(motif_attribution_sums.items(), key=lambda item: item[1])
        )
        json.dump(motif_attribution_sums, f, indent=4)
    with open(OUT_EXTRACTED_INTERACTIVE_EFFECTS, "w") as f:
        # as json cannot write frozenset as keys
        motif_pairs_interactions_sums = {
            ",".join(sorted(list(k))): v
            for k, v in motif_pairs_interactions_sums.items()
        }
        # sort according to interaction values
        motif_pairs_interactions_sums = dict(
            sorted(motif_pairs_interactions_sums.items(), key=lambda item: item[1])
        )
        json.dump(motif_pairs_interactions_sums, f, indent=4)

    # also save deltas for integrated gradients and integrated hessians
    with open(
        OUT_EXTRACTED_ADDITIVE_EFFECTS.with_stem(
            f"{OUT_EXTRACTED_ADDITIVE_EFFECTS.stem}_deltas"
        ).with_suffix(".json"),
        "w",
    ) as f:
        json.dump(ig_delta.tolist(), f, indent=4)

    with open(
        OUT_EXTRACTED_INTERACTIVE_EFFECTS.with_stem(
            f"{OUT_EXTRACTED_INTERACTIVE_EFFECTS.stem}_deltas"
        ).with_suffix(".json"),
        "w",
    ) as f:
        json.dump(ih_delta.tolist(), f, indent=4)


if __name__ == "__main__":
    main()
