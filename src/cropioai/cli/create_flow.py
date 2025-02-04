from pathlib import Path

import click

from cropioai.telemetry import Telemetry


def create_flow(name):
    """Create a new flow."""
    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    click.secho(f"Creating flow {folder_name}...", fg="green", bold=True)

    project_root = Path(folder_name)
    if project_root.exists():
        click.secho(f"Error: Folder {folder_name} already exists.", fg="red")
        return

    # Initialize telemetry
    telemetry = Telemetry()
    telemetry.flow_creation_span(class_name)

    # Create directory structure
    (project_root / "src" / folder_name).mkdir(parents=True)
    (project_root / "src" / folder_name / "cropios").mkdir(parents=True)
    (project_root / "src" / folder_name / "tools").mkdir(parents=True)
    (project_root / "tests").mkdir(exist_ok=True)

    # Create .env file
    with open(project_root / ".env", "w") as file:
        file.write("OPENAI_API_KEY=YOUR_API_KEY")

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "flow"

    # List of template files to copy
    root_template_files = [".gitignore", "pyproject.toml", "README.md"]
    src_template_files = ["__init__.py", "main.py"]
    tools_template_files = ["tools/__init__.py", "tools/custom_tool.py"]

    cropio_folders = [
        "poem_cropio",
    ]

    def process_file(src_file, dst_file):
        if src_file.suffix in [".pyc", ".pyo", ".pyd"]:
            return

        try:
            with open(src_file, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as e:
            click.secho(f"Error processing file {src_file}: {e}", fg="red")
            return

        content = content.replace("{{name}}", name)
        content = content.replace("{{flow_name}}", class_name)
        content = content.replace("{{folder_name}}", folder_name)

        with open(dst_file, "w") as file:
            file.write(content)

    # Copy and process root template files
    for file_name in root_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / file_name
        process_file(src_file, dst_file)

    # Copy and process src template files
    for file_name in src_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    # Copy tools files
    for file_name in tools_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    # Copy cropio folders
    for cropio_folder in cropio_folders:
        src_cropio_folder = templates_dir / "cropios" / cropio_folder
        dst_cropio_folder = project_root / "src" / folder_name / "cropios" / cropio_folder
        if src_cropio_folder.exists():
            for src_file in src_cropio_folder.rglob("*"):
                if src_file.is_file():
                    relative_path = src_file.relative_to(src_cropio_folder)
                    dst_file = dst_cropio_folder / relative_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    process_file(src_file, dst_file)
        else:
            click.secho(
                f"Warning: Cropio folder {cropio_folder} not found in template.",
                fg="yellow",
            )

    click.secho(f"Flow {name} created successfully!", fg="green", bold=True)
