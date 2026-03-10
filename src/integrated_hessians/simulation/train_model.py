from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from integrated_hessians.simulation import SimulatedSequence
from torch.utils.data import random_split


class MotifInteractionsDataset(Dataset):
    data: list[SimulatedSequence]

    def __init__(self, input: Path | str):
        assert input.is_file()
        assert input.name.endswith(".json")
        with open(input, "r") as f:
            json_content: list[dict] = json.load(f)
        data: list[SimulatedSequence] = [SimulatedSequence.from_dict(x) for x in json_content]
        self.data = data
        assert self.data[0].one_hot.shape == (100, 4)

    def __getitem__(self, idx):
        datapoint = self.data[idx]
        sequence = datapoint.one_hot
        phenotype = datapoint.phenotype
        return sequence, phenotype

    def __len__(self):
        return len(self.data)


class CNNEmbedding(nn.Module):
    """
    Two convolutional layers that embed a one-hot encoded sequence into a
    dense feature representation.

    Input shape:  (B, seq_len, 4)   — one-hot over 4 nucleotides
    Output shape: (B, embed_dim)    — flat feature vector

    The first conv layer learns local motif detectors (receptive field = kernel_size1).
    The second conv layer combines those motifs over a larger window (kernel_size2).
    Global average pooling then collapses the position dimension so the MLP
    receives a fixed-size vector regardless of sequence length.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv1_out: int = 64,
        conv2_out: int = 128,
        kernel_size1: int = 8,
        kernel_size2: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            # Conv 1 — motif detection
            nn.Conv1d(in_channels, conv1_out, kernel_size=kernel_size1, padding=kernel_size1 // 2),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Conv 2 — motif combination
            nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size2, padding=kernel_size2 // 2),
            nn.BatchNorm1d(conv2_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embed_dim = conv2_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 4)  ->  (B, 4, seq_len)  for Conv1d
        x = x.permute(0, 2, 1)
        x = self.cnn(x)          # (B, conv2_out, seq_len')
        x = x.mean(dim=-1)       # global average pool -> (B, conv2_out)
        return x


class SimpleModel(nn.Module):
    """
    CNN embedding followed by a depth-adjustable MLP.

    Args:
        seq_len:      Length of the input sequence (default 100).
        in_channels:  Alphabet size / one-hot width (default 4).
        conv1_out:    Filters in the first CNN layer.
        conv2_out:    Filters in the second CNN layer (= MLP input size).
        kernel_size1: Kernel size for the first CNN layer.
        kernel_size2: Kernel size for the second CNN layer.
        hidden_dim:   Width of every hidden MLP layer.
        num_layers:   Number of hidden Linear→ReLU blocks in the MLP.
                      Set to 0 for a single linear projection (embed → output).
        output_dim:   Number of output units (1 for scalar regression).
        dropout:      Dropout rate applied inside the CNN and between MLP layers.
    """

    def __init__(
        self,
        seq_len: int = 100,
        in_channels: int = 4,
        conv1_out: int = 64,
        conv2_out: int = 128,
        kernel_size1: int = 8,
        kernel_size2: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- CNN embedding ---
        self.embedding = CNNEmbedding(
            in_channels=in_channels,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            kernel_size1=kernel_size1,
            kernel_size2=kernel_size2,
            dropout=dropout,
        )
        embed_dim = self.embedding.embed_dim  # = conv2_out

        # --- MLP with adjustable depth ---
        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(num_layers):
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for samples, labels in loader:
        samples = samples.to(device, dtype=torch.float32)
        labels  = labels.to(device,  dtype=torch.float32).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(samples)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for samples, labels in loader:
            samples = samples.to(device, dtype=torch.float32)
            labels  = labels.to(device,  dtype=torch.float32).unsqueeze(1)

            outputs = model(samples)
            total_loss += criterion(outputs, labels).item() * len(samples)

            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(loader.dataset)

    ss_res = ((all_labels - all_preds) ** 2).sum()
    ss_tot = ((all_labels - all_labels.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item()

    mae = (all_labels - all_preds).abs().mean().item()

    return avg_loss, r2, mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Config ---
    BATCH_SIZE = 1000
    EPOCHS     = 200
    LR         = 1e-4
    INPUT      = Path("data/100k.json")

    # Model hyperparameters — tweak freely
    CONV1_OUT    = 100     # filters in CNN layer 1
    CONV2_OUT    = 200    # filters in CNN layer 2  (= MLP input size)
    KERNEL1      = 10      # kernel size for CNN layer 1
    KERNEL2      = 10      # kernel size for CNN layer 2
    HIDDEN_DIM   = 1000    # width of each MLP hidden layer
    NUM_LAYERS   = 8      # number of hidden MLP blocks  ← adjust depth here
    DROPOUT      = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MotifInteractionsDataset(input=INPUT)

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    model = SimpleModel(
        seq_len=100,
        in_channels=4,
        conv1_out=CONV1_OUT,
        conv2_out=CONV2_OUT,
        kernel_size1=KERNEL1,
        kernel_size2=KERNEL2,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=1,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_r2, val_mae = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "data/model_best.pth")
            print(f"New best model saved (val_loss={val_loss:.4f})")

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val R²: {val_r2:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

    model.load_state_dict(torch.load("data/model_best.pth"))
    torch.save(model.state_dict(), "data/model.pth")
    print(f"\nBest model (val_loss={best_val_loss:.4f}) saved to data/model.pth")
