from torch import nn
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype


class CNNMLP(nn.Module):
    """CNN + Flatten + MLP"""

    def __init__(
        self,
        sequence_length=50,
        alphabet_size=4,  # ACGT
        dropout: float = 0.1,
        width_multiplier=10,
        kernel_size=15,
        padding=7,
    ):
        super().__init__()

        self.sequence_length = sequence_length

        self.embedding = nn.Sequential(
            nn.Conv1d(
                alphabet_size, width_multiplier, 15, padding=7, padding_mode="reflect"
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                width_multiplier, width_multiplier, 9, padding=4, padding_mode="reflect"
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                width_multiplier, width_multiplier, 9, padding=4, padding_mode="reflect"
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(
                sequence_length * width_multiplier, sequence_length * width_multiplier
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                sequence_length * width_multiplier, sequence_length * width_multiplier
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence_length * width_multiplier, 1),
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
