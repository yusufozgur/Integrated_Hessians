from integrated_hessians.simulation import (
    Motif,
    PhenotypeStrategy,
    extract_motifs_from_jaspar_psm_file,
    SimulatedSequence,
)
import json
from integrated_hessians.simulation.simple_simulation.config import (
    MOTIFS_FILE,
    SEQLEN,
    TRAIN_DATA,
    TEST_DATA,
)


def main():
    for OUTPUT, NUM_OF_SEQUENCES in (
        (TRAIN_DATA, 10**5),
        (TEST_DATA, 10**3),
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


class SimplePhenotypeStrategy(PhenotypeStrategy):
    def get_phenotype_contribution(
        self, current_motif: Motif, other_motif: Motif
    ) -> float:
        if "Interactive" in current_motif.name and "Interactive" in other_motif.name:
            return 0.5
        else:
            return 0


if __name__ == "__main__":
    main()
