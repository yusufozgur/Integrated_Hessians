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


def simulate_random_additive_and_interactive_values(
    TRAIN_DATA,
    TRAIN_DATA_SIZE,
    TEST_DATA,
    TEST_DATA_SIZE,
    MOTIFS_FILE,
    SEQLEN,
    OUT_ADDITIVE_DEFINED_EFFECTS,
    OUT_INTERACTIVE_DEFINED_EFFECTS,
):
    TRAIN_DATA_SIZE, TEST_DATA_SIZE = int(TRAIN_DATA_SIZE), int(TEST_DATA_SIZE)
    TRAIN_DATA = Path(TRAIN_DATA)
    TEST_DATA = Path(TEST_DATA)
    OUT_ADDITIVE_DEFINED_EFFECTS = Path(OUT_ADDITIVE_DEFINED_EFFECTS)

    motifs: list[Motif] = extract_motifs_from_jaspar_psm_file(
        jaspar_pfm_file=MOTIFS_FILE
    )
    names = [m.name for m in motifs]

    def gen_additive_effects(name):
        if "Random" in name:
            return 0
        return random.uniform(0, 1)

    def gen_interactive_effects(namepair):
        if "Random" in namepair[0]:
            return 0
        if "Random" in namepair[1]:
            return 0
        return random.uniform(0, 1)

    additive_effects = {name: gen_additive_effects(name) for name in names}
    interactive_effects = {
        namepair: gen_interactive_effects(namepair)
        for namepair in list(itertools.combinations(names, 2))
    }
    OUT_ADDITIVE_DEFINED_EFFECTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ADDITIVE_DEFINED_EFFECTS, "w") as f:
        json.dump(additive_effects, f, indent=1)
    with open(OUT_INTERACTIVE_DEFINED_EFFECTS, "w") as f:
        interactive_effects_jsonable = {
            "_".join(k): v for k, v in interactive_effects.items()
        }
        json.dump(interactive_effects_jsonable, f, indent=1)

    for OUTPUT, NUM_OF_SEQUENCES in (
        (TRAIN_DATA, TRAIN_DATA_SIZE),
        (TEST_DATA, TEST_DATA_SIZE),
    ):
        sequences = [
            SimulatedSequence.from_motifs(
                motif_pool=motifs,
                length=SEQLEN,
                phenotype_strategy=RandomizedAdditive_And_Interactive(
                    additive_effects=additive_effects,
                    interactive_effects=interactive_effects,
                ),
            )
            for _ in range(NUM_OF_SEQUENCES)
        ]
        sequences_dict = [x.to_dict() for x in sequences]
        OUTPUT.parent.mkdir(exist_ok=True, parents=True)
        with open(OUTPUT, "w") as f:
            json.dump(sequences_dict, f, indent=4)


class RandomizedAdditive_And_Interactive(PhenotypeStrategy):
    def __init__(
        self,
        additive_effects: dict[str, float],
        interactive_effects: dict[tuple[str, str], float],
    ):
        self.additive_effects = additive_effects
        self.interactive_effects = interactive_effects

    def get_phenotype_contribution(
        self, current_motif: Motif, other_motif: Motif
    ) -> float:

        additive_component = self.additive_effects[current_motif.name]

        # perform an unordered key lookup
        key = (current_motif.name, other_motif.name)
        reverse_key = (other_motif.name, current_motif.name)
        interactive_component = self.interactive_effects.get(
            key
        ) or self.interactive_effects.get(reverse_key, 0)

        added_phenotype_by_current_motif = additive_component + interactive_component

        return added_phenotype_by_current_motif


def main():
    config = sys.argv[1]

    with open(config, "r") as f:
        config = json.load(f)

    simulate_random_additive_and_interactive_values(
        TRAIN_DATA=config["TRAIN_DATA"],
        TRAIN_DATA_SIZE=config["TRAIN_DATA_SIZE"],
        TEST_DATA=config["TEST_DATA"],
        TEST_DATA_SIZE=config["TEST_DATA_SIZE"],
        MOTIFS_FILE=config["MOTIFS_FILE"],
        SEQLEN=config["SEQLEN"],
        OUT_ADDITIVE_DEFINED_EFFECTS=config["OUT_ADDITIVE_DEFINED_EFFECTS"],
        OUT_INTERACTIVE_DEFINED_EFFECTS=config["OUT_INTERACTIVE_DEFINED_EFFECTS"],
    )


if __name__ == "__main__":
    main()
