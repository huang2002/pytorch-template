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


class IterationRecord(NamedTuple):
    train_avg_loss: float
    eval_avg_loss: float
    eval_accuracy: float


if __name__ == "__main__":

    device = torch.device("cuda:0")
    torch.set_default_device(device)

    dataloader_train, dataloader_eval = get_dataloaders(device)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3)
    epochs = 8

    records: list[IterationRecord] = []

    for epoch_index in range(epochs):

        print(f"[ Epoch #{epoch_index + 1} ]")

        train_result = train_loop(
            model=model,
            dataloader=dataloader_train,
            criterion=criterion,
            optimizer=optimizer,
            print_info=False,
        )

        eval_result = eval_loop(
            model=model,
            dataloader=dataloader_train,
            criterion=criterion,
            print_info=True,
        )

        records.append(
            IterationRecord(
                train_avg_loss=(
                    sum(train_result.loss_history)
                    / len(train_result.loss_history)
                ),
                eval_avg_loss=eval_result.avg_loss,
                eval_accuracy=eval_result.accuracy,
            )
        )

        print()

    print("Saving results...")

    df_records = pd.DataFrame(records)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = LOG_FILE_NAME_TEMPLATE.format(timestamp=timestamp)
    log_path = os.path.join(LOG_DIR_PATH, log_file_name)
    os.makedirs(LOG_DIR_PATH, exist_ok=True)
    df_records.to_csv(log_path)

    print("Done.")
