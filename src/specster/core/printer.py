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


class SilentDummy:
    """A class to just swallow any attrs."""

    def print(self, *args, **kwargs):
        """Prints something"""
        pass

    def rule(self, *args, **kwargs):
        """Create rule on screen"""
        pass

    @contextmanager
    def screen(self):
        """Return dummy object for screen"""
        yield self


@contextmanager
def program_render(console, title="", path="", supress_output=False):
    """Render the output of a program."""
    if specster.settings.ci or supress_output:  # do nothing on CI
        yield SilentDummy(), None
    else:
        with console.screen() as screen:
            yield console, screen
        if title:
            console.print()
            console.rule(f"[bold red]Finished command ({path}): {title}")


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
