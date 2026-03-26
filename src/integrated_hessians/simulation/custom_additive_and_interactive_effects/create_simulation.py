from integrated_hessians.simulation import (
    PhenotypeStrategy,
    SimulationMotif,
    extract_motifs_from_jaspar_psm_file,
    SimulatedSequence,
    MotifType,
)
import json
from integrated_hessians.simulation.custom_additive_and_interactive_effects.config import (
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
        roles = {
            "Motif1": MotifType.PURE_INTERACTION,
            "Interactive2": MotifType.PURE_INTERACTION,
            "Random1": MotifType.NEUTRAL,
            "Random2": MotifType.NEUTRAL,
        }
        motifs = [SimulationMotif.from_motif(m, role=roles[m.name]) for m in motifs]
        sequences = [
            SimulatedSequence.from_motifs(
                motif_pool=motifs,
                length=SEQLEN,
                phenotype_strategy=CustomAdditiveAndInteractive(),
            )
            for _ in range(NUM_OF_SEQUENCES)
        ]
        sequences_dict = [x.to_dict() for x in sequences]
        OUTPUT.parent.mkdir(exist_ok=True, parents=True)
        with open(OUTPUT, "w") as f:
            json.dump(sequences_dict, f, indent=4)


class CustomAdditiveAndInteractive(PhenotypeStrategy):
    def get_phenotype_contribution(
        self, current_motif: SimulationMotif, other_motif: SimulationMotif
    ) -> float:
        match current_motif.role:
            case MotifType.PURE_ADDITIVE:
                return 0.5
            case MotifType.HYBRID:
                contribution = 0.25
                if other_motif.role == MotifType.HYBRID:
                    contribution += 0.25
                return contribution
            case MotifType.PURE_INTERACTION:
                match other_motif.role:
                    case MotifType.PURE_INTERACTION:
                        return 0.5
                    case _:
                        return 0
            case MotifType.NEUTRAL:
                return 0


if __name__ == "__main__":
    main()
