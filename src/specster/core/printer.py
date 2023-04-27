"""
Module for printing things to the screen.
"""
from contextlib import contextmanager

from rich.console import Console

import specster
from specster.exceptions import SpecFEMError

console = Console()
consolestderr = Console(stderr=True)

stderr_style = "bold red"
stdout_style = "bold blue"


@contextmanager
def program_render(console, title="", supress_output=False):
    """Render the output of a program."""

    if specster.settings.ci or supress_output:  # do nothing on CI
        yield
    else:
        with console.screen() as screen:
            yield screen
        console.print()
        console.rule(f"[bold red]Finished command: {title}")


def print_output_run(output_dict):
    """Print the status of the specfem output."""
    if output_dict["stderr"]:
        consolestderr.print(
            f"Command {output_dict['command']} failed! STDERR:",
            style=stderr_style + " on #D3D3D3",
            justify="center",
        )
        stderr_str = "\n".join(output_dict["stderr"])
        consolestderr.print(stderr_str)

    console.print(
        "STDOUT",
        style=stdout_style + " on #D3D3D3",
        justify="center",
    )
    stdout = "\n".join(output_dict["stdout"])
    console.print(stdout)
    if output_dict["stderr"]:
        raise SpecFEMError("\n".join(output_dict["stderr"]))
