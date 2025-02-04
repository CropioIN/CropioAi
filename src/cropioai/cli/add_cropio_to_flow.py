from pathlib import Path

import click

from cropioai.cli.utils import copy_template


def add_cropio_to_flow(cropio_name: str) -> None:
    """Add a new cropio to the current flow."""
    # Check if pyproject.toml exists in the current directory
    if not Path("pyproject.toml").exists():
        print("This command must be run from the root of a flow project.")
        raise click.ClickException(
            "This command must be run from the root of a flow project."
        )

    # Determine the flow folder based on the current directory
    flow_folder = Path.cwd()
    cropios_folder = flow_folder / "src" / flow_folder.name / "cropios"

    if not cropios_folder.exists():
        print("Cropios folder does not exist in the current flow.")
        raise click.ClickException("Cropios folder does not exist in the current flow.")

    # Create the cropio within the flow's cropios directory
    create_embedded_cropio(cropio_name, parent_folder=cropios_folder)

    click.echo(
        f"Cropio {cropio_name} added to the current flow successfully!",
    )


def create_embedded_cropio(cropio_name: str, parent_folder: Path) -> None:
    """Create a new cropio within an existing flow project."""
    folder_name = cropio_name.replace(" ", "_").replace("-", "_").lower()
    class_name = cropio_name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    cropio_folder = parent_folder / folder_name

    if cropio_folder.exists():
        if not click.confirm(
            f"Cropio {folder_name} already exists. Do you want to override it?"
        ):
            click.secho("Operation cancelled.", fg="yellow")
            return
        click.secho(f"Overriding cropio {folder_name}...", fg="green", bold=True)
    else:
        click.secho(f"Creating cropio {folder_name}...", fg="green", bold=True)
        cropio_folder.mkdir(parents=True)

    # Create config and cropio.py files
    config_folder = cropio_folder / "config"
    config_folder.mkdir(exist_ok=True)

    templates_dir = Path(__file__).parent / "templates" / "cropio"
    config_template_files = ["agents.yaml", "tasks.yaml"]
    cropio_template_file = f"{folder_name}.py"  # Updated file name

    for file_name in config_template_files:
        src_file = templates_dir / "config" / file_name
        dst_file = config_folder / file_name
        copy_template(src_file, dst_file, cropio_name, class_name, folder_name)

    src_file = templates_dir / "cropio.py"
    dst_file = cropio_folder / cropio_template_file
    copy_template(src_file, dst_file, cropio_name, class_name, folder_name)

    click.secho(
        f"Cropio {cropio_name} added to the flow successfully!", fg="green", bold=True
    )
