import subprocess

import click


def install_cropio(proxy_options: list[str]) -> None:
    """
    Install the cropio by running the UV command to lock and install.
    """
    try:
        command = ["uv", "sync"] + proxy_options
        subprocess.run(command, check=True, capture_output=False, text=True)

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while running the cropio: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
