#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.sgd import SGD

from data import get_dataloaders
from model import Model
from src import eval_loop, train_loop

PROJECT_ROOT = Path(__file__).parent
LOG_ROOT = PROJECT_ROOT / "logs"


class StatisticsRecord(NamedTuple):
    train_time_seconds: float
    eval_time_seconds: float
    epoch_time_seconds: float
    train_avg_loss: float
    eval_avg_loss: float
    eval_accuracy: float


@click.group()
def main():
    pass


@main.command()
@click.option(
    "-e",
    "--epochs",
    type=int,
    prompt=True,
    help="Epochs to run."
)
@click.option(
    "-l",
    "--lr",
    type=float,
    prompt=True,
    help="Learning rate."
)
@click.option(
    "-d",
    "--device",
    "device_name",
    default="cuda:0",
    show_default=True,
    help="Device to use.",
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
    help="Save results to file or not.",
)
def train(
    *,
    epochs: int,
    lr: float,
    device_name: str,
    save: bool,
) -> None:

    device = torch.device(device_name)
    torch.set_default_device(device)

    dataloader_train, dataloader_eval, n_classes = get_dataloaders(device)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    ######## init logging ########

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir_path = LOG_ROOT / timestamp
    log_dir_relative_path = str(
        log_dir_path.relative_to(PROJECT_ROOT).as_posix()
    )

    if save:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        print(f'Results will be saved to "{log_dir_relative_path!s}".')
    else:
        print("Results won't be saved to file.")
    print()

    if save:
        config_path = log_dir_path / "config.json"
        with config_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                {
                    "criterion": repr(criterion),
                    "optimizer": repr(optimizer),
                    "epochs": epochs,
                },
                json_file,
                indent=4,
            )

    ######## train-eval ########

    statistics_records: list[StatisticsRecord] = []
    epoch_indices: list[int] = []
    confusion_matrices: list[np.ndarray] = []

    for epoch in range(1, epochs + 1):

        print(f"[ Epoch #{epoch} ]")

        begin_time = time.time()

        train_result = train_loop(
            model=model,
            dataloader=dataloader_train,
            criterion=criterion,
            optimizer=optimizer,
            print_info=False,
        )

        end_time_train = time.time()

        eval_result = eval_loop(
            model=model,
            dataloader=dataloader_eval,
            criterion=criterion,
            n_classes=n_classes,
            compute_confusion_matrix=True,
            print_info=True,
        )

        end_time_eval = time.time()
        train_time_seconds = end_time_train - begin_time
        eval_time_seconds = end_time_eval - end_time_train
        epoch_time_seconds = end_time_eval - begin_time

        if save:
            statistics_records.append(
                StatisticsRecord(
                    train_time_seconds=train_time_seconds,
                    eval_time_seconds=eval_time_seconds,
                    epoch_time_seconds=epoch_time_seconds,
                    train_avg_loss=(
                        sum(train_result.loss_history)
                        / len(train_result.loss_history)
                    ),
                    eval_avg_loss=eval_result.avg_loss,
                    eval_accuracy=eval_result.accuracy,
                )
            )

            epoch_indices.extend([epoch] * eval_result.sample_count)
            confusion_matrices.append(eval_result.confusion_matrix)

        print(f"train_time_seconds: {train_time_seconds:7.2f}")
        print(f"eval_time_seconds:  {eval_time_seconds:7.2f}")
        print(f"epoch_time_seconds: {epoch_time_seconds:7.2f}")
        print()

    ######## save results ########

    if save:

        df_statistics = pd.DataFrame(statistics_records)
        df_statistics.to_csv(log_dir_path / "statistics.csv")

        confusion_matrices_path = log_dir_path / "confusion_matrices.json"
        with confusion_matrices_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                [array.tolist() for array in confusion_matrices],
                json_file,
            )

    print("Done.")


if __name__ == "__main__":
    main()
