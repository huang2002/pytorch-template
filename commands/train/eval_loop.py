from typing import NamedTuple, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

__all__ = [
    "eval_loop",
]


class EvalResult(NamedTuple):
    sample_count: int
    confusion_matrix: np.ndarray
    accuracy: float
    avg_loss: float


def eval_loop(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    n_classes: int,
    compute_confusion_matrix: bool = False,
    print_info: bool = True,
) -> EvalResult:

    sample_count = len(cast(MNIST, dataloader.dataset))
    loss_sum: float = 0
    correct_count: int = 0
    confusion_matrix = np.zeros(
        (n_classes, n_classes),
        dtype="int64",
    )

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:

            output: torch.Tensor = model(X)

            loss: torch.Tensor = criterion(output, y)
            loss_sum += loss.item()

            pred: torch.Tensor = output.argmax(1)
            correct_count += (pred == y).type(torch.int).sum().item()

            if compute_confusion_matrix:
                for ground_truth, prediction in zip(y, pred):
                    confusion_matrix[ground_truth, prediction] += 1

    accuracy = correct_count / sample_count
    batch_count = len(dataloader)
    avg_loss = loss_sum / batch_count

    if print_info:
        print(f"Accuracy: {accuracy:.2%} | Avg Loss: {avg_loss:.6f}")

    return EvalResult(
        sample_count=sample_count,
        accuracy=accuracy,
        avg_loss=avg_loss,
        confusion_matrix=confusion_matrix,
    )
