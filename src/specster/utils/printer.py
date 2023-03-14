"""
Module for printing things to the screen.
"""
from rich.console import Console

from specster.exceptions import SpecFEMError

console = Console()
consolestderr = Console(stderr=True)

stderr_style = "bold red"
stdout_style = "bold blue"


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
