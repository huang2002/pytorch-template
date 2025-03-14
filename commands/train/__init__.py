from typing import TypedDict

import click


class TrainOptions(TypedDict):
    model: str
    dataset: str
    epochs: int
    lr: float
    batch_size: int
    eval_period: int
    device_name: str
    save_statistics: bool
    save_weights: bool


@click.command()
@click.option("-m", "--model", prompt=True, help="Model name.")
@click.option("-d", "--dataset", prompt=True, help="Dataset name.")
@click.option("-e", "--epochs", type=int, prompt=True, help="Epochs to run.")
@click.option("-l", "--lr", type=float, prompt=True, help="Learning rate.")
@click.option("-b", "--batch-size", type=int, prompt=True, help="Batch size.")
@click.option(
    "-p",
    "--eval-period",
    type=int,
    prompt=True,
    help="Period of evaluation.",
)
@click.option(
    "--device",
    "device_name",
    default="cuda:0",
    show_default=True,
    help="Device to use.",
)
@click.option(
    "--save-statistics/--no-save-statistics",
    default=True,
    show_default=True,
    help="Save statistics to file or not.",
)
@click.option(
    "--save-weights/--no-save-weights",
    default=True,
    show_default=True,
    help="Save weights to file or not.",
)
def train(**options) -> None:
    """Start model training."""

    import json
    import time
    from typing import TYPE_CHECKING, cast

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.optim.sgd import SGD

    from data import get_data
    from models import get_model
    from notebooks import copy_notebooks

    from .eval_loop import eval_loop
    from .train_loop import train_loop
    from .utils import (LOG_ROOT, PROJECT_ROOT, ConfusionMatrixRecord,
                        StatisticsRecord)

    if TYPE_CHECKING:
        options = cast(TrainOptions, options)

    device = torch.device(options["device_name"])
    torch.set_default_device(device)

    data = get_data(
        options["dataset"],
        device=device,
        batch_size=options["batch_size"],
    )
    dataloader_train = data.dataloader_train
    dataloader_eval = data.dataloader_test
    n_classes = data.n_classes

    Model = get_model(options["model"])
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=options["lr"])

    ######## init logging ########

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir_path = LOG_ROOT / timestamp
    log_dir_relative_path = str(
        log_dir_path.relative_to(PROJECT_ROOT).as_posix()
    )

    print()
    if options["save_statistics"]:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        print(f'Results will be saved to "{log_dir_relative_path!s}".')
    else:
        print("Results won't be saved to file.")
    print()

    if options["save_statistics"]:
        config_path = log_dir_path / "config.json"
        with config_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                {
                    "options": options,
                    "criterion": repr(criterion),
                    "optimizer": repr(optimizer),
                },
                json_file,
                indent=4,
            )

    ######## train-eval ########

    statistics_records: list[StatisticsRecord] = []
    epoch_indices: list[int] = []
    confusion_matrix_records: list[ConfusionMatrixRecord] = []

    total_epochs = options["epochs"]
    for epoch in range(1, total_epochs + 1):

        print(f"[ Epoch {epoch}/{total_epochs} ]")

        begin_time = time.time()

        train_result = train_loop(
            model=model,
            dataloader=dataloader_train,
            criterion=criterion,
            optimizer=optimizer,
            print_info=False,
        )

        end_time_train = time.time()

        if epoch % options["eval_period"] == 0:
            eval_result = eval_loop(
                model=model,
                dataloader=dataloader_eval,
                criterion=criterion,
                n_classes=n_classes,
                compute_confusion_matrix=True,
                print_info=True,
            )
        else:
            eval_result = None

        end_time_eval = time.time()
        train_time_seconds = end_time_train - begin_time
        eval_time_seconds = end_time_eval - end_time_train
        epoch_time_seconds = end_time_eval - begin_time

        if options["save_statistics"]:

            statistics_records.append(
                StatisticsRecord(
                    epoch=epoch,
                    train_time_seconds=train_time_seconds,
                    eval_time_seconds=eval_time_seconds,
                    epoch_time_seconds=epoch_time_seconds,
                    train_avg_loss=(
                        sum(train_result.loss_history)
                        / len(train_result.loss_history)
                    ),
                    eval_avg_loss=(
                        eval_result.avg_loss
                        if eval_result is not None
                        else np.nan
                    ),
                    eval_accuracy=(
                        eval_result.accuracy
                        if eval_result is not None
                        else np.nan
                    ),
                )
            )

            if eval_result is not None:
                epoch_indices.extend([epoch] * eval_result.sample_count)
                confusion_matrix_records.append(
                    ConfusionMatrixRecord(
                        epoch=epoch,
                        confusion_matrix=eval_result.confusion_matrix.tolist(),
                    )
                )

        print(f"train_time_seconds: {train_time_seconds:7.2f}")
        print(f"eval_time_seconds:  {eval_time_seconds:7.2f}")
        print(f"epoch_time_seconds: {epoch_time_seconds:7.2f}")
        print()

    ######## save results ########

    if options["save_statistics"]:

        df_statistics = pd.DataFrame(statistics_records)
        df_statistics.to_csv(log_dir_path / "statistics.csv")

        confusion_matrices_path = log_dir_path / "confusion_matrices.json"
        with confusion_matrices_path.open("w", encoding="utf-8") as json_file:
            json.dump(confusion_matrix_records, json_file)

        copy_notebooks(log_dir_path, model_name=options["model"])

    if options["save_weights"]:
        model_save_path = log_dir_path / "weights.pth"
        torch.save(model.state_dict(), model_save_path)

    print("Done.")
