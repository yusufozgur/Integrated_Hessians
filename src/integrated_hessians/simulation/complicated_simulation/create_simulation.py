from integrated_hessians.simulation import SimulationMotif, extract_motifs_from_jaspar_psm_file, SimulatedSequence, MotifType
from pathlib import Path
import json
def main():
    for OUTPUT, NUM_OF_SEQUENCES in ((Path("data/1M.json"),10**6),(Path("data/1k_test.json"),10**3)):
        MOTIFS_FILE = Path("src/integrated_hessians/simulation/20260304121842_JASPAR2026_combined_matrices_1525959.pfm")
        MOTIF_ROLES = Path("data/preset_motif_roles.json")

        motifs = extract_motifs_from_jaspar_psm_file(MOTIFS_FILE)
        with open(MOTIF_ROLES, "r") as f:
            motif_names_and_roles: dict[str, str] = json.load(f)
        motifs = [SimulationMotif.from_motif(m, role=MotifType[motif_names_and_roles[m.name]]) for m in motifs]
        sequences = [SimulatedSequence.from_motifs(motifs) for _ in range(NUM_OF_SEQUENCES)]
        sequences_dict = [x.to_dict() for x in sequences]
        OUTPUT.parent.mkdir(exist_ok=True, parents=True)
        with open(OUTPUT, "w") as f:
            json.dump(sequences_dict, f, indent = 4)

if __name__ == "__main__":
    main()
