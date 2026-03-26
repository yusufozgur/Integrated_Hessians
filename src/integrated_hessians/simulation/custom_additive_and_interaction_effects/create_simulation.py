from integrated_hessians.simulation import (
    PhenotypeStrategy,
    SimulationMotif,
    extract_motifs_from_jaspar_psm_file,
    SimulatedSequence,
    MotifType,
)
from pathlib import Path
import json


MOTIFS_FILE = Path(
    "src/integrated_hessians/simulation/simple_simulation/simple_motifs.pfm"
)
SEQLEN = 50


def main():
    for OUTPUT, NUM_OF_SEQUENCES in (
        (Path("data/simple_simulation/100k.json"), 10**5),
        (Path("data/simple_simulation/1k_test.json"), 10**3),
    ):
        motifs = extract_motifs_from_jaspar_psm_file(MOTIFS_FILE)
        roles = {
            "Interactive1": MotifType.PURE_INTERACTION,
            "Interactive2": MotifType.PURE_INTERACTION,
            "Random1": MotifType.NEUTRAL,
            "Random2": MotifType.NEUTRAL,
        }
        motifs = [SimulationMotif.from_motif(m, role=roles[m.name]) for m in motifs]
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
    @staticmethod
    def get_phenotype_contribution(
        current_motif: SimulationMotif, other_motif: SimulationMotif
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
