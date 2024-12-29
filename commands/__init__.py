from importlib import import_module
from pathlib import Path
from typing import Generator

import click


def get_commands() -> Generator[click.Command, None, None]:

    command_dir = Path(__file__).parent
    assert command_dir.exists() and command_dir.is_dir()

    for entry in command_dir.iterdir():

        if entry.name.startswith("_"):  # ignored entry
            continue

        if entry.is_file():
            if entry.suffix != ".py":  # non-python file
                continue
        else:  # entry.is_dir()
            if not (entry / "__init__.py").exists():  # not a module
                continue

        module_name = entry.stem
        module = import_module(f"commands.{module_name}")
        yield getattr(module, module_name)
