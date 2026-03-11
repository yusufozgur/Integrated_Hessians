import torch
import torch.nn as nn

class CNNEmbedding(nn.Module):
    """
    A stack of num_cnn_layers convolutional layers that embed a one-hot
    encoded sequence into a dense feature representation.

    Input shape:  (B, seq_len, 4)   — one-hot over 4 nucleotides
    Output shape: (B, embed_dim)    — flat feature vector

    All CNN layers share the same number of filters (cnn_out) and kernel size
    (kernel_size). The first layer accepts in_channels (4); all subsequent
    layers accept cnn_out channels. Global average pooling collapses the
    position dimension so the MLP receives a fixed-size vector regardless of
    sequence length.
    """

    def __init__(
        self,
        in_channels: int = 4,
        cnn_out: int = 128,
        kernel_size: int = 8,
        num_cnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_cnn_layers >= 1, "num_cnn_layers must be at least 1"

        layers: list[nn.Module] = []
        for i in range(num_cnn_layers):
            in_ch = in_channels if i == 0 else cnn_out
            layers += [
                nn.Conv1d(in_ch, cnn_out, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(cnn_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        self.cnn = nn.Sequential(*layers)
        self.embed_dim = cnn_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 4)  ->  (B, 4, seq_len)  for Conv1d
        x = x.permute(0, 2, 1)
        x = self.cnn(x)       # (B, cnn_out, seq_len')
        x = x.mean(dim=-1)    # global average pool -> (B, cnn_out)
        return x


class SimpleModel(nn.Module):
    """
    CNN embedding followed by a depth-adjustable MLP.

    Args:
        seq_len:        Length of the input sequence (default 100).
        in_channels:    Alphabet size / one-hot width (default 4).
        cnn_out:        Number of filters in every CNN layer.
        kernel_size:    Kernel size shared across all CNN layers.
        num_cnn_layers: Number of Conv→BN→ReLU→Dropout blocks.
        hidden_dim:     Width of every hidden MLP layer.
        num_mlp_layers: Number of hidden Linear→LN→ReLU→Dropout blocks.
                        Set to 0 for a single linear projection (embed → output).
        output_dim:     Number of output units (1 for scalar regression).
        dropout:        Dropout rate applied inside CNN and between MLP layers.
    """

    def __init__(
        self,
        seq_len: int = 100,
        in_channels: int = 4,
        cnn_out: int = 128,
        kernel_size: int = 8,
        num_cnn_layers: int = 2,
        hidden_dim: int = 512,
        num_mlp_layers: int = 4,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- CNN embedding ---
        self.embedding = CNNEmbedding(
            in_channels=in_channels,
            cnn_out=cnn_out,
            kernel_size=kernel_size,
            num_cnn_layers=num_cnn_layers,
            dropout=dropout,
        )
        embed_dim = self.embedding.embed_dim  # = cnn_out

        # --- MLP with adjustable depth ---
        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(num_mlp_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 4)
        embed = self.embedding(x)   # (B, embed_dim)
        return self.mlp(embed)      # (B, output_dim)


def get_model()->SimpleModel:
    # CNN hyperparameters
    CNN_OUT        = 128   # filters shared across all CNN layers
    KERNEL_SIZE    = 10    # kernel size shared across all CNN layers
    NUM_CNN_LAYERS = 8     # number of Conv→BN→ReLU→Dropout blocks  ← adjust depth here

    # MLP hyperparameters
    HIDDEN_DIM      = 1000  # width of each MLP hidden layer
    NUM_MLP_LAYERS  = 6     # number of hidden MLP blocks  ← adjust depth here

    DROPOUT = 0.1

    return SimpleModel(
        seq_len=100,
        in_channels=4,
        cnn_out=CNN_OUT,
        kernel_size=KERNEL_SIZE,
        num_cnn_layers=NUM_CNN_LAYERS,
        hidden_dim=HIDDEN_DIM,
        num_mlp_layers=NUM_MLP_LAYERS,
        output_dim=1,
        dropout=DROPOUT,
    )
