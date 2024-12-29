from typing import NamedTuple

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

__all__ = [
    "train_loop",
]


class TrainResult(NamedTuple):
    loss_history: list[float]


def train_loop(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    print_info: bool = True,
) -> TrainResult:

    batch_count = len(dataloader)

    loss_history: list[float] = []

    model.train()
    for batch_index, (X, y) in enumerate(dataloader):

        pred: torch.Tensor = model(X)
        loss: torch.Tensor = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if print_info:
            print(
                f"Batch {batch_index + 1:>2d}/{batch_count} "
                f"-- loss: {loss_history[-1]:.6f}"
            )

    return TrainResult(
        loss_history=loss_history,
    )
