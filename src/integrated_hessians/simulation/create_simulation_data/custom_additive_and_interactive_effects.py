import itertools
import json
from pathlib import Path
import random
import sys

from integrated_hessians.simulation import (
    Motif,
    PhenotypeStrategy,
    SimulatedSequence,
    extract_motifs_from_jaspar_psm_file,
)


def simulate_custom_additive_and_interactive_values(
    TRAIN_DATA, TRAIN_DATA_SIZE, TEST_DATA, TEST_DATA_SIZE, MOTIFS_FILE, SEQLEN
):

    TRAIN_DATA_SIZE, TEST_DATA_SIZE = int(TRAIN_DATA_SIZE), int(TEST_DATA_SIZE)
    TRAIN_DATA = Path(TRAIN_DATA)
    TEST_DATA = Path(TEST_DATA)

    motifs = extract_motifs_from_jaspar_psm_file(jaspar_pfm_file=MOTIFS_FILE)

    for OUTPUT, NUM_OF_SEQUENCES in (
        (TRAIN_DATA, TRAIN_DATA_SIZE),
        (TEST_DATA, TEST_DATA_SIZE),
    ):
        sequences = [
            SimulatedSequence.from_motifs(
                motif_pool=motifs,
                length=SEQLEN,
                phenotype_strategy=Additive_And_Interactive(),
            )
            for _ in range(NUM_OF_SEQUENCES)
        ]
        sequences_dict = [x.to_dict() for x in sequences]
        OUTPUT.parent.mkdir(exist_ok=True, parents=True)
        with open(OUTPUT, "w") as f:
            json.dump(sequences_dict, f, indent=4)


class Additive_And_Interactive(PhenotypeStrategy):
    def __init__(self):
        # half interaction: 1,2
        # full interaction: 2,3
        # half additive: 4
        # full additive: 5
        # half interaction half additive: 6, 6,1
        self.additive_effects = {
            "no_1_weak_interactive": 0,
            "no_2_dynamic_interactive": 0,
            "no_3_strong_interactive": 0,
            "no_4_weak_additive": 0.25,
            "no_5_strong_additive": 0.5,
            "no_6_weak_interactive_weak_additive": 0.25,
            "no_7_random": 0,
            "no_8_random": 0,
        }
        self.interactive_effects = {
            # weak interaction
            ("no_1_weak_interactive", "no_2_dynamic_interactive"): 0.25,
            ("Motif2", "no_1_weak_interactive"): 0.25,
            # strong interaction
            ("no_2_dynamic_interactive", "no_3_strong_interactive"): 0.5,
            ("no_3_strong_interactive", "no_2_dynamic_interactive"): 0.5,
            # weak interaction with hybrid
            ("no_1_weak_interactive", "no_6_weak_interactive_weak_additive"): 0.25,
            ("no_6_weak_interactive_weak_additive", "no_1_weak_interactive"): 0.25,
        }

    def get_phenotype_contribution(
        self, current_motif: Motif, other_motif: Motif
    ) -> float:

        additive_component = self.additive_effects[current_motif.name]

        interactive_component = self.interactive_effects.get(
            (current_motif.name, other_motif.name), 0
        )

        added_phenotype_by_current_motif = additive_component + interactive_component

        return added_phenotype_by_current_motif


def main():
    config = sys.argv[1]

    with open(config, "r") as f:
        config = json.load(f)

    simulate_custom_additive_and_interactive_values(
        TRAIN_DATA=config["TRAIN_DATA"],
        TRAIN_DATA_SIZE=config["TRAIN_DATA_SIZE"],
        TEST_DATA=config["TEST_DATA"],
        TEST_DATA_SIZE=config["TEST_DATA_SIZE"],
        MOTIFS_FILE=config["MOTIFS_FILE"],
        SEQLEN=config["SEQLEN"],
    )


if __name__ == "__main__":
    main()
