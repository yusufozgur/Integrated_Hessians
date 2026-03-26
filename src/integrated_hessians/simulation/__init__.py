# needed for self type reference in SimulationMotif
from __future__ import annotations

from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
from numpy.typing import NDArray
from enum import Enum, auto
import random
from typing import NewType, Tuple, Optional
import jaxtyping as jx
from abc import ABC, abstractmethod

Nucleotide_Sequence = str
NUCLEOTIDE_ORDER = ["A", "C", "G", "T"]


@dataclass
class Motif:
    name: str
    # matrix shapes: (4, length)
    count_matrix: NDArray
    probability_matrix: NDArray

    def sample(self) -> Nucleotide_Sequence:
        """
        Samples a realized motif from the probability matrix.
        """
        sequence = []
        for position_probs in np.permute_dims(self.probability_matrix, (1, 0)):
            nucleotide = random.choices(NUCLEOTIDE_ORDER, weights=position_probs, k=1)[
                0
            ]
            sequence.append(nucleotide)
        sequence = "".join(sequence)
        return sequence


def extract_motifs_from_jaspar_psm_file(jaspar_pfm_file: Path) -> list[Motif]:
    with open(jaspar_pfm_file, "r") as f:
        motifs_file_content = f.read()

    motifs = motifs_file_content.split(">")
    # there is an initial "" element
    motifs = [m for m in motifs if m != ""]

    if len(motifs) % 4 != 0 and len(motifs) > 0:
        raise Exception(
            f"Number of motifs in the jaspar pfm file must be a positive multiple of 4, and currently there are {len(motifs)} motifs inside the file."
        )

    motifs = [m.strip().split("\n") for m in motifs]
    motifs = [[row.strip() for row in m] for m in motifs]
    motifs = [{"name": m[0], "pos_freq_matrix": m[1:]} for m in motifs]
    for m in motifs:
        m["pos_freq_matrix"] = [row.split() for row in m["pos_freq_matrix"]]
        m["pos_freq_matrix"] = [[float(x) for x in row] for row in m["pos_freq_matrix"]]
        m["pos_freq_matrix"] = np.array(m["pos_freq_matrix"])
        # convert pfm, where there are counts to ppm, where there will be probabilities
        column_sums = m["pos_freq_matrix"].sum(axis=0)
        m["pos_prob_matrix"] = m["pos_freq_matrix"] / column_sums

    return [
        Motif(
            name=m["name"],
            count_matrix=m["pos_freq_matrix"],
            probability_matrix=m["pos_prob_matrix"],
        )
        for m in motifs
    ]


class MotifType(Enum):
    """
    -   Group A: Purely additive. Existence of a single motif from this group increases the arbitrary phenotype value by 0.5.
    -   Group B: Hybrid. Existence of a single motif from this group increses the arbitrary phenotype value by 0.25, if another motif of this group exists in the same sequence, it is instead increased by 0.5. Which means that if two motifs from group B is present in a sequence, the value is 1.
    -   Group C: Purely interactive. Arbitrary phenotype is 1 if two motifs from this group exists in the sequence, otherwise phenotype is 0.
    -   Group D: Neutral, these motifs do not affect the phenotype.
    """

    PURE_ADDITIVE = auto()
    HYBRID = auto()
    PURE_INTERACTION = auto()
    NEUTRAL = auto()

    def to_json(self):
        return self.name

    @classmethod
    def from_json(cls, value):
        return cls[value]


@dataclass
class SimulationMotif(Motif):
    """
    A motif with a role in the simulation
    """

    role: MotifType

    @staticmethod
    def from_motif(motif: Motif, role: Optional[MotifType] = None) -> SimulationMotif:
        if not role:
            role = random.choice(list(MotifType))
        return SimulationMotif(
            name=motif.name,
            count_matrix=motif.count_matrix,
            probability_matrix=motif.probability_matrix,
            role=role,
        )


class PhenotypeStrategy(ABC):
    @abstractmethod
    def get_phenotype_contribution(
        self, current_motif: SimulationMotif, other_motif: SimulationMotif
    ) -> float:
        pass


@dataclass
class SimulatedSequence:
    length: int
    nucleotides: Nucleotide_Sequence
    one_hot: jx.Float[NDArray[np.float32], "sequence_length alphabet_length"]
    phenotype: float
    motif_names: Tuple[str, str]
    motif_types: Tuple[MotifType, MotifType]
    motif_mask_1: str
    motif_mask_2: str

    @staticmethod
    def from_motifs(
        motif_pool: list[SimulationMotif],
        length: int,
        phenotype_strategy: PhenotypeStrategy,
    ):

        nucleotides = "".join(random.choices(NUCLEOTIDE_ORDER, k=length))

        MOTIF_COUNT = 2
        motifs = random.choices(motif_pool, k=MOTIF_COUNT)

        phenotype = 0.0
        nucleotides, motif_mask_1 = SimulatedSequence.insert_motif(
            nucleotides, motifs[0]
        )

        phenotype += phenotype_strategy.get_phenotype_contribution(
            current_motif=motifs[0], other_motif=motifs[1]
        )

        nucleotides, motif_mask_2 = SimulatedSequence.insert_motif(
            nucleotides, motifs[1]
        )
        phenotype += phenotype_strategy.get_phenotype_contribution(
            current_motif=motifs[1], other_motif=motifs[0]
        )

        motif_names = [m.name for m in motifs]
        motif_types = [m.role for m in motifs]
        one_hot = SimulatedSequence.encode_one_hot(nucleotides)
        assert len(motif_names) == 2
        motif_names = (motif_names[0], motif_names[1])
        assert len(motif_types) == 2
        motif_types = (motif_types[0], motif_types[1])

        return SimulatedSequence(
            length=length,
            nucleotides=nucleotides,
            one_hot=one_hot,
            phenotype=phenotype,
            motif_names=motif_names,
            motif_types=motif_types,
            motif_mask_1=motif_mask_1,
            motif_mask_2=motif_mask_2,
        )

    @staticmethod
    def insert_motif(
        nucleotides: Nucleotide_Sequence, motif: Motif
    ) -> Tuple[Nucleotide_Sequence, str]:
        motif_sequence = motif.sample()
        motif_length = len(motif_sequence)
        sequence_length = len(nucleotides)

        # Calculate valid insertion positions
        max_position = sequence_length - motif_length
        if max_position < 0:
            raise ValueError("Motif is longer than the sequence")

        # Random insertion position
        insertion_pos = random.randint(0, max_position)

        # Insert motif into sequence
        nucleotides = (
            nucleotides[:insertion_pos]
            + motif_sequence
            + nucleotides[insertion_pos + motif_length :]
        )

        # Create mask: 0 for background, 1 for motif positions
        mask = (
            "0" * insertion_pos
            + "1" * motif_length
            + "0" * (sequence_length - insertion_pos - motif_length)
        )

        assert len(nucleotides) == sequence_length
        assert len(mask) == sequence_length

        return (nucleotides, mask)

    @staticmethod
    def encode_one_hot(seqn: Nucleotide_Sequence) -> NDArray:
        mapping = {nuc: i for i, nuc in enumerate(NUCLEOTIDE_ORDER)}
        indices = [mapping[nuc] for nuc in seqn]
        one_hot = np.zeros((len(seqn), 4))
        one_hot[np.arange(len(seqn)), indices] = 1
        return one_hot

    def to_dict(self) -> dict:
        # Convert to dict first
        data = asdict(self)
        # Convert numpy array to list
        del data["one_hot"]
        # Convert Enum members to their names (strings)
        data["motif_types"] = [mt.name for mt in self.motif_types]
        return data

    @staticmethod
    def from_dict(data: dict) -> SimulatedSequence:
        data = data.copy()
        data["one_hot"] = SimulatedSequence.encode_one_hot(data["nucleotides"])
        data["motif_names"] = tuple(data["motif_names"])
        # Convert strings back to Enum members using the name
        data["motif_types"] = tuple(MotifType[mt] for mt in data["motif_types"])
        return SimulatedSequence(**data)
