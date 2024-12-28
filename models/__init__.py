from importlib import import_module

from torch import nn


def get_model(module_name: str) -> nn.Module:
    module = import_module(f"models.{module_name}")
    return getattr(module, "Model")
