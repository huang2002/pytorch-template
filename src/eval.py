from typing import NamedTuple, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

__all__ = [
    "eval_loop",
]


class EvalResult(NamedTuple):
    accuracy: float
    avg_loss: float


def eval_loop(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    print_info: bool = True,
) -> EvalResult:

    loss_sum: float = 0
    correct_count: int = 0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            pred: torch.Tensor = model(X)
            loss: torch.Tensor = criterion(pred, y)
            loss_sum += loss.item()
            correct_count += (pred.argmax(1) == y).type(torch.int).sum().item()

    dataset_size = len(cast(MNIST, dataloader.dataset))
    accuracy = correct_count / dataset_size
    batch_count = len(dataloader)
    avg_loss = loss_sum / batch_count

    if print_info:
        print(f"Accuracy: {accuracy:.2%} | Avg Loss: {avg_loss:.6f}")

    return EvalResult(
        accuracy=accuracy,
        avg_loss=avg_loss,
    )
