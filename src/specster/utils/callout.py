"""
A module for calling out to specfem programs.
"""
import subprocess
from time import sleep, time_ns


def command_run_and_stream(command, cwd=None):
    """Run a command and poll to print input/output"""
    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=cwd,
    )
    # Read stdout from subprocess until the buffer is empty !
    for line in iter(p.stdout.readline, b""):
        yield "stdout", line
    for line in iter(p.stderr.readline, b""):
        yield "stderr", line
    # This ensures the process has completed, AND sets the 'returncode' attr
    while p.poll() is None:
        sleep(0.1)  # Don't waste CPU-cycles
    if p.poll() is not None:
        yield "return_code", p.poll()
        return


def run_command(command, cwd=None, print_=True):
    """Run command, capture and print stderr and stdout."""
    _captured = {"stdout": [], "stderr": [], "return_code": []}
    time_start = time_ns()
    if print_:
        print(command)
        print("-" * len(str(command)))
    for key, value in command_run_and_stream(command, cwd=cwd):
        if key != "return_code":
            formatted = value.decode("UTF8").strip()
            _captured[key].append(formatted)
            if print_:
                print(formatted)
        else:
            _captured["return_code"] = value
    _captured["time_elapsed"] = (time_ns() - time_start) / 1_000_000_000
    return _captured
