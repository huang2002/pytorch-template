
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor

from .._shared import DATA_ROOT_PATH, GetDataResult

__all__ = [
    "get_data",
]


def get_data(
    *,
    device: torch.device,
    batch_size: int,
) -> GetDataResult:

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

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

    return GetDataResult(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        n_classes=len(dataset_test.classes),
    )
