import os
import time
from typing import NamedTuple

import pandas as pd
import torch
from torch import nn
from torch.optim.sgd import SGD

from data import get_dataloaders
from model import Model
from src import eval_loop, train_loop

LOG_DIR_PATH = os.path.join(os.path.dirname(__file__), "logs/")
LOG_FILE_NAME_TEMPLATE = "{timestamp}.csv"

device = torch.device("cuda:0")


class IterationRecord(NamedTuple):
    train_time_seconds: float
    eval_time_seconds: float
    epoch_time_seconds: float
    train_avg_loss: float
    eval_avg_loss: float
    eval_accuracy: float


if __name__ == "__main__":

    torch.set_default_device(device)

    dataloader_train, dataloader_eval = get_dataloaders(device)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3)
    epochs = 8

    records: list[IterationRecord] = []

    for epoch_index in range(epochs):

        print(f"[ Epoch #{epoch_index + 1} ]")

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
            print_info=True,
        )

        end_time_eval = time.time()
        train_time_seconds = end_time_train - begin_time
        eval_time_seconds = end_time_eval - end_time_train
        epoch_time_seconds = end_time_eval - begin_time

        records.append(
            IterationRecord(
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

        print(f"train_time_seconds: {train_time_seconds:7.2f}")
        print(f"eval_time_seconds:  {eval_time_seconds:7.2f}")
        print(f"epoch_time_seconds: {epoch_time_seconds:7.2f}")
        print()

    print("Saving results...")

    df_records = pd.DataFrame(records)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = LOG_FILE_NAME_TEMPLATE.format(timestamp=timestamp)
    log_path = os.path.join(LOG_DIR_PATH, log_file_name)
    os.makedirs(LOG_DIR_PATH, exist_ok=True)
    df_records.to_csv(log_path)

    print("Done.")
