import math
import torch
from torch import nn
from typing import Literal


# from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(
                d_model
            )
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class TransformerEncoder(nn.Module):
    """ """

    def __init__(
        self,
        rna_length=100,
        alphabet_size=4,  # ACGT
        d_model=512,
        nhead=8,
        num_layers=8,
        dropout: float = 0.1,
        mode: Literal[
            "predict_phenotype", "return_latent_representations"
        ] = "predict_phenotype",
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.mode = mode

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        positional_encoding = positionalencoding1d(d_model, rna_length)
        self.register_buffer(
            "positional_encoding", positional_encoding, persistent=False
        )

        self.embedding = nn.Sequential(
            nn.Conv1d(alphabet_size, d_model, 15, padding=7, padding_mode="reflect"),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Conv1d(d_model, d_model, 15, padding=7, padding_mode="reflect"),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Conv1d(d_model, d_model, 15, padding=7, padding_mode="reflect"),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Conv1d(d_model, d_model, 15, padding=7, padding_mode="reflect"),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Conv1d(d_model, d_model, 15, padding=7, padding_mode="reflect"),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # no relu for rich embedding
            nn.Conv1d(d_model, d_model, 9, padding=4, padding_mode="reflect"),
            nn.BatchNorm1d(d_model),
            nn.Dropout(p=dropout),
        )

        self.head = nn.Linear(d_model, 1)

        self.pos_dropout = nn.Dropout(p=dropout)
        self.output_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x.shape = (batchsize, rna_length, alphabet_size)
        x = x.permute(0, 2, 1)
        # x.shape = (batchsize, alphabet_size, rna_length)
        x = self.embedding(x)
        # x.shape = (batchsize, d_model, rna_length)
        x = x.permute(0, 2, 1)
        # x.shape = (batchsize, rna_length, d_model)
        x = self.pos_dropout(x + self.positional_encoding)
        x = self.transformer_encoder(x)
        # x.shape = (batchsize, rna_length, d_model)

        if self.mode == "return_latent_representations":
            return x

        x = self.output_dropout(x)
        # Mean pool over sequence length
        x = x.mean(dim=1)
        # x.shape = (batchsize, d_model)
        x = self.head(x)
        # x.shape = (batchsize, 1)
        return x


def get_model(
        rna_length=100,
        alphabet_size=4,
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout = 0.2,
        ) -> TransformerEncoder:
    return TransformerEncoder(
        rna_length=rna_length,
        alphabet_size=alphabet_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )
