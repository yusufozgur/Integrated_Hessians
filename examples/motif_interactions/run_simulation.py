#run via uv run python -m examples.motif_interactions.run_simulation
from examples.motif_interactions import SimulationMotif, extract_motifs_from_jaspar_psm_file, SimulatedSequence
from pathlib import Path
import json

def main():
    MOTIFS_FILE = "examples/motif_interactions/20260304121842_JASPAR2026_combined_matrices_1525959.pfm"
    NUM_OF_SEQUENCES = 10**3
    OUTPUT = Path("data/1k.json")

    motifs = extract_motifs_from_jaspar_psm_file(Path(MOTIFS_FILE))
    motifs = [SimulationMotif.from_motif(m) for m in motifs]
    
    sequences = [SimulatedSequence.from_motifs(motifs) for _ in range(NUM_OF_SEQUENCES)]
    sequences_dict = [x.to_dict() for x in sequences]
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT, "w") as f:
        json.dump(sequences_dict, f, indent = 4)

if __name__ == "__main__":
    main()
