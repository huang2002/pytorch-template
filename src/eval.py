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
    predictions: np.ndarray
    ground_truths: np.ndarray
    accuracy: float
    avg_loss: float


def eval_loop(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    keep_pred_and_truth: bool = False,
    print_info: bool = True,
) -> EvalResult:

    pred_list = []
    truth_list = []
    loss_sum: float = 0
    correct_count: int = 0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:

            output: torch.Tensor = model(X)

            loss: torch.Tensor = criterion(output, y)
            loss_sum += loss.item()

            pred: torch.Tensor = output.argmax(1)
            correct_count += (pred == y).type(torch.int).sum().item()

            if keep_pred_and_truth:
                pred_list.extend(pred.numpy())
                truth_list.extend(y.numpy())

    dataset_size = len(cast(MNIST, dataloader.dataset))
    accuracy = correct_count / dataset_size
    batch_count = len(dataloader)
    avg_loss = loss_sum / batch_count

    if print_info:
        print(f"Accuracy: {accuracy:.2%} | Avg Loss: {avg_loss:.6f}")

    return EvalResult(
        accuracy=accuracy,
        avg_loss=avg_loss,
        predictions=np.array(pred_list),
        ground_truths=np.array(truth_list),
    )
