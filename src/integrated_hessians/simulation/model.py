import math
import torch
from torch import nn
from torch.nn import functional as F
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
        d_model=1024,
        nhead=16,
        num_layers=5,
        mode: Literal[
            "predict_phenotype", "return_latent_representations"
        ] = "predict_phenotype",
    ):
        super().__init__()
        # RNA size is assumed 100

        assert d_model % nhead == 0

        self.mode = mode

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        positional_encoding = positionalencoding1d(d_model, rna_length)
        # you need to register positional encodings as buffer(non-trainable) for
        #   automatic moving into device
        self.register_buffer(
            "positional_encoding", positional_encoding, persistent=False
        )

        self.embedding = nn.Sequential(
            nn.Conv1d(alphabet_size, d_model, 15, padding=7, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 9, padding=4, padding_mode="reflect"),
            nn.BatchNorm1d(d_model),
        )

        # this is equivalent to 1x1 conv as we are passing a 3D tensor
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x.shape = (batchsize, alphabet_size, rna_length)

        # embedder projects local information into higher dimensional representations
        x = self.embedding(x)
        # x.shape = (batchsize, d_model, rna_length)

        # we change dimensional order as transformer expects to have latent dimension at the end
        x = x.permute(0, 2, 1)
        # x.shape = (batchsize, rna_length, d_model)

        # add positional encodings
        # you may ask why add instead of concat: https://stats.stackexchange.com/questions/586813/intuitive-explanation-for-summing-the-embedding-and-positional-encoding-in-the-t
        x = x + self.positional_encoding

        # encoder learns to output latent representations
        x = self.transformer_encoder(x)
        # x.shape = (batchsize, rna_length, d_model)

        if self.mode == "return_latent_representations":
            return x

        # the head learns to extract phenotype values from latent representations
        x = self.head(x)
        # x.shape = (batchsize, rna_length, 1)

        x = x[:, :, 0]
        # x.shape = (batchsize, rna_length)

        # as our data should be between 0-1 (or even narrower)
        x = F.sigmoid(x)

        return x


def get_model() -> TransformerEncoder:

    return TransformerEncoder()
