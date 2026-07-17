"""Tests for cli.py — Click CLI commands."""

from __future__ import annotations

import pytest

click = pytest.importorskip("click", reason="click is required for CLI tests")
CliRunner = pytest.importorskip("click.testing", reason="click.testing is required for CLI tests").CliRunner

from open_mlpipe.cli import main  # noqa: E402


@pytest.mark.unit
def test_run_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--data" in result.output


@pytest.mark.unit
def test_profile_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(main, ["profile", "--help"])
    assert result.exit_code == 0
    assert "--data" in result.output


@pytest.mark.unit
def test_main_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0


@pytest.mark.unit
def test_run_with_temp_csv(temp_csv: str):
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--data", temp_csv, "--project", "test-cli-project"])
    # Level 1 should succeed or fail gracefully (not crash)
    assert result.exit_code == 0
