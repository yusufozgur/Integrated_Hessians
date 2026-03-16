import math
import torch
from torch import nn
from typing import Literal, Annotated
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype

class CNNDense(nn.Module):
    """ """

    def __init__(
        self,
        sequence_length=50,
        alphabet_size=4,  # ACGT
        dropout: float = 0.2,
        width_multiplier = 300,
        kernel_size = 15,
        padding = 7
    ):
        super().__init__()

        self.sequence_length = sequence_length

        self.embedding = nn.Sequential(
            # seq length 50
            nn.Conv1d(alphabet_size, width_multiplier, kernel_size, padding=padding, padding_mode="reflect"),
            nn.BatchNorm1d(width_multiplier),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(2),

            # 25
            nn.Conv1d(width_multiplier, width_multiplier*2, kernel_size, padding=padding, padding_mode="reflect"),
            nn.BatchNorm1d(width_multiplier*2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(2),

            # 12
            nn.Flatten(),
            # 6*width_multiplier*4
            nn.Linear(12*width_multiplier*2, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier*4),
            nn.Linear(width_multiplier*4, width_multiplier),
            nn.Linear(width_multiplier, 1),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batchsize rna_length alphabet_size"])-> Float[Tensor, "batchsize 1"]:
        x: Float[Tensor, "batchsize alphabet_size rna_length"] = x.permute(0, 2, 1)
        x: Float[Tensor, "batchsize 1"] = self.embedding(x)
        return x
