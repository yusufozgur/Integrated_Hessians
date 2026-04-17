from math import ceil

from torch import nn
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype


class CNNMLP(nn.Module):
    """CNN + Flatten + MLP"""

    def __init__(
        self,
        sequence_length=100,
        alphabet_size=4,  # ACGT
        dropout: float = 0.1,
        width_multiplier=10,
        kernel_size=15,
        padding=7,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        linear_layer_width = ceil(sequence_length / 4) * width_multiplier

        self.embedding = nn.Sequential(
            nn.Conv1d(
                alphabet_size, width_multiplier, 15, padding=7, padding_mode="zeros"
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                width_multiplier, width_multiplier, 9, padding=4, padding_mode="zeros"
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                width_multiplier, width_multiplier, 9, padding=4, padding_mode="zeros"
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(linear_layer_width, linear_layer_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linear_layer_width, linear_layer_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linear_layer_width, 1),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batchsize sequence_length alphabet_size"]
    ) -> Float[Tensor, "batchsize 1"]:
        x_permuted: Float[Tensor, "batchsize alphabet_size sequence_length"] = (
            x.permute(0, 2, 1)
        )
        x_embedded: Float[Tensor, "batchsize 1"] = self.embedding(x_permuted)
        return x_embedded
