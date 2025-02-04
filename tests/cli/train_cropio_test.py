import subprocess
from unittest import mock

from cropioai.cli.train_cropio import train_cropio


@mock.patch("cropioai.cli.train_cropio.subprocess.run")
def test_train_cropio_positive_iterations(mock_subprocess_run):
    n_iterations = 5
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "train", str(n_iterations)],
        returncode=0,
        stdout="Success",
        stderr="",
    )

    train_cropio(n_iterations, "trained_agents_data.pkl")

    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "train", str(n_iterations), "trained_agents_data.pkl"],
        capture_output=False,
        text=True,
        check=True,
    )


@mock.patch("cropioai.cli.train_cropio.click")
def test_train_cropio_zero_iterations(click):
    train_cropio(0, "trained_agents_data.pkl")
    click.echo.assert_called_once_with(
        "An unexpected error occurred: The number of iterations must be a positive integer.",
        err=True,
    )


@mock.patch("cropioai.cli.train_cropio.click")
def test_train_cropio_negative_iterations(click):
    train_cropio(-2, "trained_agents_data.pkl")
    click.echo.assert_called_once_with(
        "An unexpected error occurred: The number of iterations must be a positive integer.",
        err=True,
    )


@mock.patch("cropioai.cli.train_cropio.click")
@mock.patch("cropioai.cli.train_cropio.subprocess.run")
def test_train_cropio_called_process_error(mock_subprocess_run, click):
    n_iterations = 5
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["uv", "run", "train", str(n_iterations)],
        output="Error",
        stderr="Some error occurred",
    )
    train_cropio(n_iterations, "trained_agents_data.pkl")

    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "train", str(n_iterations), "trained_agents_data.pkl"],
        capture_output=False,
        text=True,
        check=True,
    )
    click.echo.assert_has_calls(
        [
            mock.call.echo(
                "An error occurred while training the cropio: Command '['uv', 'run', 'train', '5']' returned non-zero exit status 1.",
                err=True,
            ),
            mock.call.echo("Error", err=True),
        ]
    )


@mock.patch("cropioai.cli.train_cropio.click")
@mock.patch("cropioai.cli.train_cropio.subprocess.run")
def test_train_cropio_unexpected_exception(mock_subprocess_run, click):
    n_iterations = 5
    mock_subprocess_run.side_effect = Exception("Unexpected error")
    train_cropio(n_iterations, "trained_agents_data.pkl")

    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "train", str(n_iterations), "trained_agents_data.pkl"],
        capture_output=False,
        text=True,
        check=True,
    )
    click.echo.assert_called_once_with(
        "An unexpected error occurred: Unexpected error", err=True
    )
