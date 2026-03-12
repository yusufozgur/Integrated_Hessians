from integrated_hessians.simulation import (
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
            "SM1": MotifType.PURE_INTERACTION,
            "SM4": MotifType.PURE_INTERACTION,
            "SM2": MotifType.NEUTRAL,
            "SM3": MotifType.NEUTRAL,
        }
        motifs = [SimulationMotif.from_motif(m, role=roles[m.name]) for m in motifs]
        sequences = [
            SimulatedSequence.from_motifs(motifs, length=SEQLEN)
            for _ in range(NUM_OF_SEQUENCES)
        ]
        sequences_dict = [x.to_dict() for x in sequences]
        OUTPUT.parent.mkdir(exist_ok=True, parents=True)
        with open(OUTPUT, "w") as f:
            json.dump(sequences_dict, f, indent=4)


if __name__ == "__main__":
    main()
