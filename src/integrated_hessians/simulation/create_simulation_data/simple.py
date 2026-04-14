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


def simulate_simple(
    TRAIN_DATA, TEST_DATA, MOTIFS_FILE, SEQLEN, TRAIN_DATA_SIZE, TEST_DATA_SIZE
):
    TRAIN_DATA_SIZE, TEST_DATA_SIZE = int(TRAIN_DATA_SIZE), int(TEST_DATA_SIZE)
    TRAIN_DATA = Path(TRAIN_DATA)
    TEST_DATA = Path(TEST_DATA)

    class SimplePhenotypeStrategy(PhenotypeStrategy):
        def get_phenotype_contribution(
            self, current_motif: Motif, other_motif: Motif
        ) -> float:
            if (
                "Interactive" in current_motif.name
                and "Interactive" in other_motif.name
            ):
                return 0.5
            else:
                return 0

    for OUTPUT, NUM_OF_SEQUENCES in (
        (TRAIN_DATA, TRAIN_DATA_SIZE),
        (TEST_DATA, TEST_DATA_SIZE),
    ):
        motifs = extract_motifs_from_jaspar_psm_file(jaspar_pfm_file=MOTIFS_FILE)
        sequences = [
            SimulatedSequence.from_motifs(
                motif_pool=motifs,
                length=SEQLEN,
                phenotype_strategy=SimplePhenotypeStrategy(),
            )
            for _ in range(NUM_OF_SEQUENCES)
        ]
        sequences_dict = [x.to_dict() for x in sequences]
        OUTPUT.parent.mkdir(exist_ok=True, parents=True)
        with open(OUTPUT, "w") as f:
            json.dump(sequences_dict, f, indent=4)


def main():
    config = sys.argv[1]

    with open(config, "r") as f:
        config = json.load(f)

    simulate_simple(
        TRAIN_DATA=config["TRAIN_DATA"],
        TRAIN_DATA_SIZE=config["TRAIN_DATA_SIZE"],
        TEST_DATA=config["TEST_DATA"],
        TEST_DATA_SIZE=config["TEST_DATA_SIZE"],
        MOTIFS_FILE=config["MOTIFS_FILE"],
        SEQLEN=config["SEQLEN"],
    )


if __name__ == "__main__":
    main()
