#!/usr/bin/env python3
from importlib import import_module
from pathlib import Path

import click


@click.group()
def main():
    pass


command_dir = Path(__file__).parent / "commands"
assert command_dir.exists() and command_dir.is_dir()
for file in command_dir.iterdir():
    if file.is_file() and file.suffix == ".py":
        module_name = file.stem
        module = import_module(f"commands.{module_name}")
        main.add_command(getattr(module, module_name))


if __name__ == "__main__":
    main()
