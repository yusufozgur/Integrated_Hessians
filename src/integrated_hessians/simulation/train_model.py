from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from integrated_hessians.simulation import SimulatedSequence
from torch.utils.data import random_split
from integrated_hessians.simulation.model import get_model

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



if __name__ == "__main__":
    # --- Config ---
    BATCH_SIZE = 1000
    EPOCHS     = 20
    LR         = 1e-4
    INPUT      = Path("data/1M.json")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MotifInteractionsDataset(input=INPUT)

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    model = get_model()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_r2, val_mae = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val R²: {val_r2:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "epoch":      epoch,
                "train_loss": round(train_loss, 4),
                "val_loss":   round(val_loss,   4),
                "val_r2":     round(val_r2,     4),
                "val_mae":    round(val_mae,     4),
            }
            torch.save(model.state_dict(), "data/model_best.pth")
            with open("data/model_best_evaluation.json", "w") as f:
                json.dump(best_metrics, f, indent=2)
            print(f"New best model saved (val_loss={val_loss:.4f})")


    model.load_state_dict(torch.load("data/model_best.pth"))
    print(f"\nBest model (val_loss={best_val_loss:.4f}) saved to data/model_best.pth")
    print(f"Best metrics: {best_metrics}")
