from importlib import import_module

from ._shared import GetDataFunction, GetDataResult


def get_data(module_name: str, **kwargs) -> GetDataResult:
    module = import_module(f"data.{module_name}")
    _get_data: GetDataFunction = getattr(module, "get_data")
    return _get_data(**kwargs)
