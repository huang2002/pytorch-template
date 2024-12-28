from pathlib import Path
from typing import NamedTuple, TypedDict

PROJECT_ROOT = Path(__file__).parent.parent
LOG_ROOT = PROJECT_ROOT / "logs"
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"


class StatisticsRecord(NamedTuple):
    epoch: int
    train_time_seconds: float
    eval_time_seconds: float
    epoch_time_seconds: float
    train_avg_loss: float
    eval_avg_loss: float
    eval_accuracy: float


class ConfusionMatrixRecord(TypedDict):
    epoch: int
    confusion_matrix: list[list[int]]
