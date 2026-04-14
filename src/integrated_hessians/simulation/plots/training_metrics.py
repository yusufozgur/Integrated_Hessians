import matplotlib.pyplot as plt

import numpy as np


def plot_training_metrics(
    title: str,
    train_epoch_losses: list[float],
    train_step_losses: list[float],
    val_epoch_losses: list[float],
    val_step_losses: list[float],
    val_r2_per_epoch: list[float],
    val_mae_per_epoch: list[float],
) -> plt.Figure:  # type: ignore
    # Extract metrics

    # Set up the visualization layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # Increased height for titles
    axes = axes.flatten()

    # 1. Step-wise Losses
    best_step_loss = min(train_step_losses) if train_step_losses else None
    if train_step_losses:
        axes[0].plot(
            train_step_losses, label="Train Step Loss", color="blue", alpha=0.4
        )
    if val_step_losses:
        axes[0].plot(
            np.linspace(0, len(train_step_losses), len(val_step_losses)),
            val_step_losses,
            "o",
            label="Val Step Loss",
            color="orange",
            markersize=2,
        )

    axes[0].set_title(
        f"Loss per Training Step\n(Best Train Step Loss: {best_step_loss:.4f})"
    )
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # 2. Epoch-wise Losses
    epochs = range(1, len(train_epoch_losses) + 1)
    best_val_loss = min(val_epoch_losses) if val_epoch_losses else None

    if train_epoch_losses:
        axes[1].plot(
            epochs,
            train_epoch_losses,
            marker="o",
            label="Train Epoch Loss",
            color="blue",
        )
    if val_epoch_losses:
        axes[1].plot(
            epochs, val_epoch_losses, marker="x", label="Val Epoch Loss", color="orange"
        )

    axes[1].set_title(f"Loss per Epoch\n(Best Val Epoch Loss: {best_val_loss:.4f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # 3. Validation R^2
    best_r2 = max(val_r2_per_epoch) if val_r2_per_epoch else None
    if val_r2_per_epoch:
        epochs_r2 = range(1, len(val_r2_per_epoch) + 1)
        axes[2].plot(
            epochs_r2, val_r2_per_epoch, marker="s", color="green", label="Val $R^2$"
        )

    axes[2].set_title(f"Validation $R^2$ Score\n(Best $R^2$: {best_r2:.4f})")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("$R^2$ Value")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.7)

    # 4. Validation MAE
    best_mae = min(val_mae_per_epoch) if val_mae_per_epoch else None
    if val_mae_per_epoch:
        epochs_mae = range(1, len(val_mae_per_epoch) + 1)
        axes[3].plot(
            epochs_mae, val_mae_per_epoch, marker="d", color="red", label="Val MAE"
        )

    axes[3].set_title(f"Validation Mean Absolute Error\n(Best MAE: {best_mae:.4f})")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("MAE")
    axes[3].legend()
    axes[3].grid(True, linestyle="--", alpha=0.7)

    fig.suptitle(title)

    fig.tight_layout()
    return fig
