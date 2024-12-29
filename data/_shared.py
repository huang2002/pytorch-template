from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
from torch.utils.data import DataLoader, Dataset

DATA_ROOT_PATH = Path(__file__).parent / "_datasets"


class GetDataResult(NamedTuple):
    dataset_train: Dataset
    dataset_test: Dataset
    dataloader_train: DataLoader
    dataloader_test: DataLoader
    n_classes: int


class GetDataFunction(Protocol):
    def __call__(
        self,
        *,
        device: torch.device,
        batch_size: int,
    ) -> GetDataResult: ...
