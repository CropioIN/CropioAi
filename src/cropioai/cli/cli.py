import os
from importlib.metadata import version as get_version
from typing import Optional, Tuple

import click

from cropioai.cli.add_cropio_to_flow import add_cropio_to_flow
from cropioai.cli.create_cropio import create_cropio
from cropioai.cli.create_flow import create_flow
from cropioai.cli.cropio_chat import run_chat
from cropioai.memory.storage.takeoff_task_outputs_storage import (
    TakeoffTaskOutputsSQLiteStorage,
)

from .authentication.main import AuthenticationCommand
from .deploy.main import DeployCommand
from .evaluate_cropio import evaluate_cropio
from .install_cropio import install_cropio
from .takeoff_flow import takeoff_flow
from .plot_flow import plot_flow
from .replay_from_task import replay_task_command
from .reset_memories_command import reset_memories_command
from .run_cropio import run_cropio
from .tools.main import ToolCommand
from .train_cropio import train_cropio
from .update_cropio import update_cropio


@click.group()
@click.version_option(get_version("cropioai"))
def cropioai():
    """Top-level command group for cropioai."""


@cropio.inmand()
@click.argument("type", type=click.Choice(["cropio", "flow"]))
@click.argument("name")
@click.option("--provider", type=str, help="The provider to use for the cropio")
@click.option("--skip_provider", is_flag=True, help="Skip provider validation")
def create(type, name, provider, skip_provider=False):
    """Create a new cropio, or flow."""
    if type == "cropio":
        create_cropio(name, provider, skip_provider)
    elif type == "flow":
        create_flow(name)
    else:
        click.secho("Error: Invalid type. Must be 'cropio' or 'flow'.", fg="red")


@cropio.inmand()
@click.option(
    "--tools", is_flag=True, help="Show the installed version of cropioai tools"
)
def version(tools):
    """Show the installed version of cropioai."""
    try:
        cropioai_version = get_version("cropioai")
    except Exception:
        cropioai_version = "unknown version"
    click.echo(f"cropioai version: {cropioai_version}")

    if tools:
        try:
            tools_version = get_version("cropioai")
            click.echo(f"cropioai tools version: {tools_version}")
        except Exception:
            click.echo("cropioai tools not installed")


@cropio.inmand()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=5,
    help="Number of iterations to train the cropio",
)
@click.option(
    "-f",
    "--filename",
    type=str,
    default="trained_agents_data.pkl",
    help="Path to a custom file for training",
)
def train(n_iterations: int, filename: str):
    """Train the cropio."""
    click.echo(f"Training the Cropio for {n_iterations} iterations")
    train_cropio(n_iterations, filename)


@cropio.inmand()
@click.option(
    "-t",
    "--task_id",
    type=str,
    help="Replay the cropio from this task ID, including all subsequent tasks.",
)
def replay(task_id: str) -> None:
    """
    Replay the cropio execution from a specific task.

    Args:
        task_id (str): The ID of the task to replay from.
    """
    try:
        click.echo(f"Replaying the cropio from task {task_id}")
        replay_task_command(task_id)
    except Exception as e:
        click.echo(f"An error occurred while replaying: {e}", err=True)


@cropio.inmand()
def log_tasks_outputs() -> None:
    """
    Retrieve your latest cropio.takeoff() task outputs.
    """
    try:
        storage = TakeoffTaskOutputsSQLiteStorage()
        tasks = storage.load()

        if not tasks:
            click.echo(
                "No task outputs found. Only cropio takeoff task outputs are logged."
            )
            return

        for index, task in enumerate(tasks, 1):
            click.echo(f"Task {index}: {task['task_id']}")
            click.echo(f"Description: {task['expected_output']}")
            click.echo("------")

    except Exception as e:
        click.echo(f"An error occurred while logging task outputs: {e}", err=True)


@cropio.inmand()
@click.option("-l", "--long", is_flag=True, help="Reset LONG TERM memory")
@click.option("-s", "--short", is_flag=True, help="Reset SHORT TERM memory")
@click.option("-e", "--entities", is_flag=True, help="Reset ENTITIES memory")
@click.option("-kn", "--knowledge", is_flag=True, help="Reset KNOWLEDGE storage")
@click.option(
    "-k",
    "--takeoff-outputs",
    is_flag=True,
    help="Reset LATEST TAKEOFF TASK OUTPUTS",
)
@click.option("-a", "--all", is_flag=True, help="Reset ALL memories")
def reset_memories(
    long: bool,
    short: bool,
    entities: bool,
    knowledge: bool,
    takeoff_outputs: bool,
    all: bool,
) -> None:
    """
    Reset the cropio memories (long, short, entity, latest_cropio_takeoff_ouputs). This will delete all the data saved.
    """
    try:
        if not all and not (long or short or entities or knowledge or takeoff_outputs):
            click.echo(
                "Please specify at least one memory type to reset using the appropriate flags."
            )
            return
        reset_memories_command(long, short, entities, knowledge, takeoff_outputs, all)
    except Exception as e:
        click.echo(f"An error occurred while resetting memories: {e}", err=True)


@cropio.inmand()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=3,
    help="Number of iterations to Test the cropio",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="gpt-4o-mini",
    help="LLM Model to run the tests on the Cropio. For now only accepting only OpenAI models.",
)
def test(n_iterations: int, model: str):
    """Test the cropio and evaluate the results."""
    click.echo(f"Testing the cropio for {n_iterations} iterations with model {model}")
    evaluate_cropio(n_iterations, model)


@cropio.inmand(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.pass_context
def install(context):
    """Install the Cropio."""
    install_cropio(context.args)


@cropio.inmand()
def run():
    """Run the Cropio."""
    click.echo("Running the Cropio")
    run_cropio()


@cropio.inmand()
def update():
    """Update the pyproject.toml of the Cropio project to use uv."""
    update_cropio()


@cropio.inmand()
def signup():
    """Sign Up/Login to CropioAI+."""
    AuthenticationCommand().signup()


@cropio.inmand()
def login():
    """Sign Up/Login to CropioAI+."""
    AuthenticationCommand().signup()


# DEPLOY CREWAI+ COMMANDS
@cropioai.group()
def deploy():
    """Deploy the Cropio CLI group."""
    pass


@cropioai.group()
def tool():
    """Tool Repository related commands."""
    pass


@deploy.command(name="create")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
def deploy_create(yes: bool):
    """Create a Cropio deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.create_cropio(yes)


@deploy.command(name="list")
def deploy_list():
    """List all deployments."""
    deploy_cmd = DeployCommand()
    deploy_cmd.list_cropios()


@deploy.command(name="push")
@click.option("-u", "--uuid", type=str, help="Cropio UUID parameter")
def deploy_push(uuid: Optional[str]):
    """Deploy the Cropio."""
    deploy_cmd = DeployCommand()
    deploy_cmd.deploy(uuid=uuid)


@deploy.command(name="status")
@click.option("-u", "--uuid", type=str, help="Cropio UUID parameter")
def deply_status(uuid: Optional[str]):
    """Get the status of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_cropio_status(uuid=uuid)


@deploy.command(name="logs")
@click.option("-u", "--uuid", type=str, help="Cropio UUID parameter")
def deploy_logs(uuid: Optional[str]):
    """Get the logs of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_cropio_logs(uuid=uuid)


@deploy.command(name="remove")
@click.option("-u", "--uuid", type=str, help="Cropio UUID parameter")
def deploy_remove(uuid: Optional[str]):
    """Remove a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.remove_cropio(uuid=uuid)


@tool.command(name="create")
@click.argument("handle")
def tool_create(handle: str):
    tool_cmd = ToolCommand()
    tool_cmd.create(handle)


@tool.command(name="install")
@click.argument("handle")
def tool_install(handle: str):
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.install(handle)


@tool.command(name="publish")
@click.option(
    "--force",
    is_flag=True,
    show_default=True,
    default=False,
    help="Bypasses Git remote validations",
)
@click.option("--public", "is_public", flag_value=True, default=False)
@click.option("--private", "is_public", flag_value=False)
def tool_publish(is_public: bool, force: bool):
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.publish(is_public, force)


@cropioai.group()
def flow():
    """Flow related commands."""
    pass


@flow.command(name="takeoff")
def flow_run():
    """Takeoff the Flow."""
    click.echo("Running the Flow")
    takeoff_flow()


@flow.command(name="plot")
def flow_plot():
    """Plot the Flow."""
    click.echo("Plotting the Flow")
    plot_flow()


@flow.command(name="add-cropio")
@click.argument("cropio_name")
def flow_add_cropio(cropio_name):
    """Add a cropio to an existing flow."""
    click.echo(f"Adding cropio {cropio_name} to the flow")
    add_cropio_to_flow(cropio_name)


@cropio.inmand()
def chat():
    """
    Start a conversation with the Cropio, collecting user-supplied inputs,
    and using the Chat LLM to generate responses.
    """
    click.secho(
        "\nStarting a conversation with the Cropio\n" "Type 'exit' or Ctrl+C to quit.\n",
    )

    run_chat()


if __name__ == "__main__":
    cropioai()
