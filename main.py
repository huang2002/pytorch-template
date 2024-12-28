#!/usr/bin/env python3
import click

from commands import get_commands


@click.group()
def main():
    pass


for command in get_commands():
    main.add_command(command)

if __name__ == "__main__":
    main()
