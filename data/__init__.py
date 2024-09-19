import os
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

__all__ = [
    "get_dataloaders",
]

DATA_ROOT_PATH = os.path.dirname(__file__)


class GetDataloadersResult(NamedTuple):
    dataloader_train: DataLoader
    dataloader_test: DataLoader


def get_dataloaders(device: torch.device) -> GetDataloadersResult:

    transform_to_tensor = ToTensor()
    transform = Lambda(lambda x: transform_to_tensor(x).to(device))

    dataset_train = datasets.MNIST(
        root=DATA_ROOT_PATH,
        train=True,
        download=True,
        transform=transform,
    )

    dataset_test = datasets.MNIST(
        root=DATA_ROOT_PATH,
        train=False,
        download=True,
        transform=transform,
    )

    dataloader_train = DataLoader(dataset_train, batch_size=64)
    dataloader_test = DataLoader(dataset_test, batch_size=64)

    return GetDataloadersResult(
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
    )
