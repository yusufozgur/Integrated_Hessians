from pathlib import Path
import sys
from polars import Unknown
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from integrated_hessians.simulation import SimulatedSequence
from torch.utils.data import random_split
from integrated_hessians.simulation.model import CNNMLP
from tqdm import tqdm
import random
from os import cpu_count
import numpy as np


def train_model(
    SEQLEN,
    TRAIN_DATA,
    BATCH_SIZE,
    LR,
    L2_WEIGHT_DECAY,
    EPOCHS,
    OUT_BEST_MODEL,
    OUT_BEST_MODEL_EVAL,
    MODEL_WIDTH_MULTIPLIER,
    EXPAND_DATA_DISTRIBUTION_ALONG_BASELINE_TO_INPUT_PATH,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MotifInteractionsDataset(
        input=TRAIN_DATA,
        SEQLEN=SEQLEN,
        EXPAND_DATA_DISTRIBUTION_ALONG_BASELINE_TO_INPUT_PATH=EXPAND_DATA_DISTRIBUTION_ALONG_BASELINE_TO_INPUT_PATH,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    num_of_cores = cpu_count()
    if num_of_cores is None:
        num_of_cores = 1
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_of_cores
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=num_of_cores
    )

    model = CNNMLP(sequence_length=SEQLEN, width_multiplier=MODEL_WIDTH_MULTIPLIER)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=L2_WEIGHT_DECAY
    )

    best_val_loss = float("inf")

    # Tracking containers
    all_train_epoch_losses = []
    all_train_step_losses = []
    all_val_epoch_losses = []
    all_val_step_losses = []
    # New tracking containers for R2 and MAE
    all_val_r2 = []
    all_val_mae = []

    for epoch in range(1, EPOCHS + 1):
        # Training
        train_loss, train_steps = train(
            model, train_loader, optimizer, criterion, device
        )
        all_train_epoch_losses.append(train_loss)
        all_train_step_losses.extend(train_steps)

        # Validation
        val_loss, val_r2, val_mae, val_steps = evaluate(
            model, val_loader, criterion, device
        )
        all_val_epoch_losses.append(val_loss)
        all_val_step_losses.extend(val_steps)
        all_val_r2.append(val_r2)
        all_val_mae.append(val_mae)

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
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_r2": round(val_r2, 4),
                "val_mae": round(val_mae, 4),
                "train_epoch_losses": all_train_epoch_losses,
                "train_step_losses": all_train_step_losses,
                "val_epoch_losses": all_val_epoch_losses,
                "val_step_losses": all_val_step_losses,
                "val_r2_per_epoch": all_val_r2,
                "val_mae_per_epoch": all_val_mae,
            }
            torch.save(model.state_dict(), OUT_BEST_MODEL)
            with open(OUT_BEST_MODEL_EVAL, "w") as f:
                json.dump(best_metrics, f, indent=2)
            print(f"New best model saved (val_loss={val_loss:.4f})")

    model.load_state_dict(torch.load(OUT_BEST_MODEL))
    print(f"\nBest model (val_loss={best_val_loss:.4f}) saved to {OUT_BEST_MODEL}")

    # Final save of the metrics for the best model state
    with open(OUT_BEST_MODEL_EVAL, "w") as f:
        json.dump(best_metrics, f, indent=2)  # pyright: ignore[reportPossiblyUnboundVariable]


class MotifInteractionsDataset(Dataset):
    data: list[SimulatedSequence]

    def __init__(
        self,
        input: Path,
        SEQLEN: int,
        EXPAND_DATA_DISTRIBUTION_ALONG_BASELINE_TO_INPUT_PATH: bool,
    ):
        assert input.is_file()
        assert input.name.endswith(".json")
        with open(input, "r") as f:
            json_content: list[dict] = json.load(f)
        data: list[SimulatedSequence] = [
            SimulatedSequence.from_dict(x) for x in json_content
        ]
        self.data = data
        assert self.data[0].one_hot.shape == (SEQLEN, 4)

        self.data_augmentation = EXPAND_DATA_DISTRIBUTION_ALONG_BASELINE_TO_INPUT_PATH

    def __getitem__(self, idx):
        datapoint: SimulatedSequence = self.data[idx]
        assert isinstance(datapoint, SimulatedSequence)
        sequence = datapoint.one_hot
        phenotype = datapoint.phenotype

        if self.data_augmentation:
            choose_augmentation = random.choices(
                ["baseline", "smooth_labels", "do_nothing"], weights=[0.1, 0.4, 0.5]
            )[0]
            match choose_augmentation:
                case "baseline":
                    sequence = np.full_like(sequence, 0.25)
                    phenotype = 0.0
                case "smooth_labels":
                    # convert the one hot encoding into a sampling along the path from full 0.25 vectors to the input
                    # this will help with out of distribution errors seen while using methods like Integrated Gradients or Integrated Hessians
                    # For example, if baseline of .25s is not 0, but 1, then the attributions values will not reflect the expectations
                    random_float_between_01_ends_not_inclusive = random.uniform(
                        1e-10, 1 - 1e-10
                    )
                    # 0 is not included as phenotype has to be made 0 then, its handled in choose_augmentation == "baseline" case
                    # 1 is not included as its the same as input, handled in "do_nothing" case
                    sequence = label_smoothing(
                        one_hot=sequence,
                        smoothing_coefficient=random_float_between_01_ends_not_inclusive,
                    )
                case "do_nothing":
                    pass
                case _:
                    raise Exception()
        return sequence, phenotype

    def __len__(self):
        return len(self.data)


def label_smoothing(one_hot, smoothing_coefficient: float):
    """
    smoothing_coefficient == 1 is all 0.25 and == 0 is equal to one_hot
    """
    n_classes = 4
    add_term = smoothing_coefficient / n_classes
    return (1 - smoothing_coefficient) * one_hot + add_term


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    step_losses = []

    for samples, labels in tqdm(loader, desc="Training"):
        samples = samples.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        step_losses.append(batch_loss)
        total_loss += batch_loss * len(samples)

    epoch_loss = total_loss / len(loader.dataset)
    return epoch_loss, step_losses


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    step_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for samples, labels in loader:
            samples = samples.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float).unsqueeze(1)

            outputs = model(samples)
            batch_loss_val = criterion(outputs, labels).item()

            step_losses.append(batch_loss_val)
            total_loss += batch_loss_val * len(samples)

            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(loader.dataset)

    ss_res = ((all_labels - all_preds) ** 2).sum()
    ss_tot = ((all_labels - all_labels.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item()

    mae = (all_labels - all_preds).abs().mean().item()

    return avg_loss, r2, mae, step_losses


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_path>")
        return

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = json.load(f)

    train_model(
        SEQLEN=config["SEQLEN"],
        TRAIN_DATA=Path(config["TRAIN_DATA"]),
        BATCH_SIZE=config["BATCH_SIZE"],
        LR=config["LR"],
        L2_WEIGHT_DECAY=config["L2_WEIGHT_DECAY"],
        EPOCHS=config["EPOCHS"],
        OUT_BEST_MODEL=config["OUT_BEST_MODEL"],
        OUT_BEST_MODEL_EVAL=config["OUT_BEST_MODEL_EVAL"],
        MODEL_WIDTH_MULTIPLIER=config["MODEL_WIDTH_MULTIPLIER"],
        EXPAND_DATA_DISTRIBUTION_ALONG_BASELINE_TO_INPUT_PATH=config[
            "EXPAND_DATA_DISTRIBUTION_ALONG_BASELINE_TO_INPUT_PATH"
        ],
    )


if __name__ == "__main__":
    main()
