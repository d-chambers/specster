"""
Tests for running commands.
"""
from specster.core.callout import run_command


class TestBasic:
    """Test running basic shell commands."""

    def test_ls(self, capsys):
        """simply list directory contents."""
        out = run_command("ls")
        assert out["stdout"]
        assert not out["stderr"]
        assert out["return_code"] == 0

    def test_bad_command(self):
        """Ensure a bad command returns a non-zero code and stderr list."""
        out = run_command("bbbbbadbbcommandls", print_=False)
        assert out["stderr"]
        assert not out["stdout"]
        assert "not found" in "".join(out["stderr"])
        assert out["return_code"] != 0
