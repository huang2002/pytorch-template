from importlib import import_module
from pathlib import Path
from typing import Generator

import click


def get_commands() -> Generator[click.Command, None, None]:

    command_dir = Path(__file__).parent
    assert command_dir.exists() and command_dir.is_dir()

    for file in command_dir.iterdir():
        if not file.is_file() or file.suffix != ".py":
            continue
        module_name = file.stem
        if module_name == "__init__":
            continue
        module = import_module(f"commands.{module_name}")
        yield getattr(module, module_name)
