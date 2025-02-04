import subprocess

import click
from packaging import version

from cropioai.cli.utils import read_toml
from cropioai.cli.version import get_cropioai_version


def run_cropio() -> None:
    """
    Run the cropio by running a command in the UV environment.
    """
    command = ["uv", "run", "run_cropio"]
    cropioai_version = get_cropioai_version()
    min_required_version = "0.71.0"

    pyproject_data = read_toml()

    if pyproject_data.get("tool", {}).get("poetry") and (
        version.parse(cropioai_version) < version.parse(min_required_version)
    ):
        click.secho(
            f"You are running an older version of cropioAI ({cropioai_version}) that uses poetry pyproject.toml. "
            f"Please run `cropioai update` to update your pyproject.toml to use uv.",
            fg="red",
        )

    try:
        subprocess.run(command, capture_output=False, text=True, check=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the cropio: {e}", err=True)
        click.echo(e.output, err=True, nl=True)

        if pyproject_data.get("tool", {}).get("poetry"):
            click.secho(
                "It's possible that you are using an old version of cropioAI that uses poetry, please run `cropioai update` to update your pyproject.toml to use uv.",
                fg="yellow",
            )

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
