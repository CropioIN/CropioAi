import subprocess
from unittest import mock

import pytest

from cropioai.cli import evaluate_cropio


@pytest.mark.parametrize(
    "n_iterations,model",
    [
        (1, "gpt-4o"),
        (5, "gpt-3.5-turbo"),
        (10, "gpt-4"),
    ],
)
@mock.patch("cropioai.cli.evaluate_cropio.subprocess.run")
def test_cropio_success(mock_subprocess_run, n_iterations, model):
    """Test the cropio function for successful execution."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=f"uv run test {n_iterations} {model}", returncode=0
    )
    result = evaluate_cropio.evaluate_cropio(n_iterations, model)

    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "test", str(n_iterations), model],
        capture_output=False,
        text=True,
        check=True,
    )
    assert result is None


@mock.patch("cropioai.cli.evaluate_cropio.click")
def test_test_cropio_zero_iterations(click):
    evaluate_cropio.evaluate_cropio(0, "gpt-4o")
    click.echo.assert_called_once_with(
        "An unexpected error occurred: The number of iterations must be a positive integer.",
        err=True,
    )


@mock.patch("cropioai.cli.evaluate_cropio.click")
def test_test_cropio_negative_iterations(click):
    evaluate_cropio.evaluate_cropio(-2, "gpt-4o")
    click.echo.assert_called_once_with(
        "An unexpected error occurred: The number of iterations must be a positive integer.",
        err=True,
    )


@mock.patch("cropioai.cli.evaluate_cropio.click")
@mock.patch("cropioai.cli.evaluate_cropio.subprocess.run")
def test_test_cropio_called_process_error(mock_subprocess_run, click):
    n_iterations = 5
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["uv", "run", "test", str(n_iterations), "gpt-4o"],
        output="Error",
        stderr="Some error occurred",
    )
    evaluate_cropio.evaluate_cropio(n_iterations, "gpt-4o")

    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "test", "5", "gpt-4o"],
        capture_output=False,
        text=True,
        check=True,
    )
    click.echo.assert_has_calls(
        [
            mock.call.echo(
                "An error occurred while testing the cropio: Command '['uv', 'run', 'test', '5', 'gpt-4o']' returned non-zero exit status 1.",
                err=True,
            ),
            mock.call.echo("Error", err=True),
        ]
    )


@mock.patch("cropioai.cli.evaluate_cropio.click")
@mock.patch("cropioai.cli.evaluate_cropio.subprocess.run")
def test_test_cropio_unexpected_exception(mock_subprocess_run, click):
    # Arrange
    n_iterations = 5
    mock_subprocess_run.side_effect = Exception("Unexpected error")
    evaluate_cropio.evaluate_cropio(n_iterations, "gpt-4o")

    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "test", "5", "gpt-4o"],
        capture_output=False,
        text=True,
        check=True,
    )
    click.echo.assert_called_once_with(
        "An unexpected error occurred: Unexpected error", err=True
    )
