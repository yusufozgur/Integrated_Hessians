"""
We need to assign random roles to motifs, but it would be nice to keep these roles the same between simulation runs, hence, this is a seperate script.
"""

from integrated_hessians.simulation import (
    SimulationMotif,
    extract_motifs_from_jaspar_psm_file,
)
from pathlib import Path
import json


def main():
    MOTIFS_FILE = "src/integrated_hessians/simulation/20260304121842_JASPAR2026_combined_matrices_1525959.pfm"
    OUTPUT = Path("data/preset_motif_roles.json")

    motifs = extract_motifs_from_jaspar_psm_file(Path(MOTIFS_FILE))
    motifs_with_roles: list[SimulationMotif] = [
        SimulationMotif.from_motif(m) for m in motifs
    ]

    motif_names_and_roles = {motif.name: motif.role.name for motif in motifs_with_roles}

    with open(OUTPUT, "w") as f:
        json.dump(motif_names_and_roles, f, indent=4)


if __name__ == "__main__":
    main()
