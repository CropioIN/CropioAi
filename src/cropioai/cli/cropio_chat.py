import json
import platform
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import tomli
from packaging import version

from cropioai.cli.utils import read_toml
from cropioai.cli.version import get_cropioai_version
from cropioai.cropio import Cropio
from cropioai.llm import LLM
from cropioai.types.cropio_chat import ChatInputField, ChatInputs
from cropioai.utilities.llm_utils import create_llm

MIN_REQUIRED_VERSION = "0.98.0"


def check_conversational_cropios_version(
    cropioai_version: str, pyproject_data: dict
) -> bool:
    """
    Check if the installed cropioAI version supports conversational cropios.

    Args:
        cropioai_version: The current version of cropioAI.
        pyproject_data: Dictionary containing pyproject.toml data.

    Returns:
        bool: True if version check passes, False otherwise.
    """
    try:
        if version.parse(cropioai_version) < version.parse(MIN_REQUIRED_VERSION):
            click.secho(
                "You are using an older version of cropioAI that doesn't support conversational cropios. "
                "Run 'uv upgrade cropioai' to get the latest version.",
                fg="red",
            )
            return False
    except version.InvalidVersion:
        click.secho("Invalid cropioAI version format detected.", fg="red")
        return False
    return True


def run_chat():
    """
    Runs an interactive chat loop using the Cropio's chat LLM with function calling.
    Incorporates cropio_name, cropio_description, and input fields to build a tool schema.
    Exits if cropio_name or cropio_description are missing.
    """
    cropioai_version = get_cropioai_version()
    pyproject_data = read_toml()

    if not check_conversational_cropios_version(cropioai_version, pyproject_data):
        return

    cropio, cropio_name = load_cropio_and_name()
    chat_llm = initialize_chat_llm(cropio)
    if not chat_llm:
        return

    # Indicate that the cropio is being analyzed
    click.secho(
        "\nAnalyzing cropio and required inputs - this may take 3 to 30 seconds "
        "depending on the complexity of your cropio.",
        fg="white",
    )

    # Start loading indicator
    loading_complete = threading.Event()
    loading_thread = threading.Thread(target=show_loading, args=(loading_complete,))
    loading_thread.start()

    try:
        cropio_chat_inputs = generate_cropio_chat_inputs(cropio, cropio_name, chat_llm)
        cropio_tool_schema = generate_cropio_tool_schema(cropio_chat_inputs)
        system_message = build_system_message(cropio_chat_inputs)

        # Call the LLM to generate the introductory message
        introductory_message = chat_llm.call(
            messages=[{"role": "system", "content": system_message}]
        )
    finally:
        # Stop loading indicator
        loading_complete.set()
        loading_thread.join()

    # Indicate that the analysis is complete
    click.secho("\nFinished analyzing cropio.\n", fg="white")

    click.secho(f"Assistant: {introductory_message}\n", fg="green")

    messages = [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": introductory_message},
    ]

    available_functions = {
        cropio_chat_inputs.cropio_name: create_tool_function(cropio, messages),
    }

    chat_loop(chat_llm, messages, cropio_tool_schema, available_functions)


def show_loading(event: threading.Event):
    """Display animated loading dots while processing."""
    while not event.is_set():
        print(".", end="", flush=True)
        time.sleep(1)
    print()


def initialize_chat_llm(cropio: Cropio) -> Optional[LLM]:
    """Initializes the chat LLM and handles exceptions."""
    try:
        return create_llm(cropio.chat_llm)
    except Exception as e:
        click.secho(
            f"Unable to find a Chat LLM. Please make sure you set chat_llm on the cropio: {e}",
            fg="red",
        )
        return None


def build_system_message(cropio_chat_inputs: ChatInputs) -> str:
    """Builds the initial system message for the chat."""
    required_fields_str = (
        ", ".join(
            f"{field.name} (desc: {field.description or 'n/a'})"
            for field in cropio_chat_inputs.inputs
        )
        or "(No required fields detected)"
    )

    return (
        "You are a helpful AI assistant for the CropioAI platform. "
        "Your primary purpose is to assist users with the cropio's specific tasks. "
        "You can answer general questions, but should guide users back to the cropio's purpose afterward. "
        "For example, after answering a general question, remind the user of your main purpose, such as generating a research report, and prompt them to specify a topic or task related to the cropio's purpose. "
        "You have a function (tool) you can call by name if you have all required inputs. "
        f"Those required inputs are: {required_fields_str}. "
        "Once you have them, call the function. "
        "Please keep your responses concise and friendly. "
        "If a user asks a question outside the cropio's scope, provide a brief answer and remind them of the cropio's purpose. "
        "After calling the tool, be prepared to take user feedback and make adjustments as needed. "
        "If you are ever unsure about a user's request or need clarification, ask the user for more information. "
        "Before doing anything else, introduce yourself with a friendly message like: 'Hey! I'm here to help you with [cropio's purpose]. Could you please provide me with [inputs] so we can get started?' "
        "For example: 'Hey! I'm here to help you with uncovering and reporting cutting-edge developments through thorough research and detailed analysis. Could you please provide me with a topic you're interested in? This will help us generate a comprehensive research report and detailed analysis.'"
        f"\nCropio Name: {cropio_chat_inputs.cropio_name}"
        f"\nCropio Description: {cropio_chat_inputs.cropio_description}"
    )


def create_tool_function(cropio: Cropio, messages: List[Dict[str, str]]) -> Any:
    """Creates a wrapper function for running the cropio tool with messages."""

    def run_cropio_tool_with_messages(**kwargs):
        return run_cropio_tool(cropio, messages, **kwargs)

    return run_cropio_tool_with_messages


def flush_input():
    """Flush any pending input from the user."""
    if platform.system() == "Windows":
        # Windows platform
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getch()
    else:
        # Unix-like platforms (Linux, macOS)
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def chat_loop(chat_llm, messages, cropio_tool_schema, available_functions):
    """Main chat loop for interacting with the user."""
    while True:
        try:
            # Flush any pending input before accepting new input
            flush_input()

            user_input = get_user_input()
            handle_user_input(
                user_input, chat_llm, messages, cropio_tool_schema, available_functions
            )

        except KeyboardInterrupt:
            click.echo("\nExiting chat. Goodbye!")
            break
        except Exception as e:
            click.secho(f"An error occurred: {e}", fg="red")
            break


def get_user_input() -> str:
    """Collect multi-line user input with exit handling."""
    click.secho(
        "\nYou (type your message below. Press 'Enter' twice when you're done):",
        fg="blue",
    )
    user_input_lines = []
    while True:
        line = input()
        if line.strip().lower() == "exit":
            return "exit"
        if line == "":
            break
        user_input_lines.append(line)
    return "\n".join(user_input_lines)


def handle_user_input(
    user_input: str,
    chat_llm: LLM,
    messages: List[Dict[str, str]],
    cropio_tool_schema: Dict[str, Any],
    available_functions: Dict[str, Any],
) -> None:
    if user_input.strip().lower() == "exit":
        click.echo("Exiting chat. Goodbye!")
        return

    if not user_input.strip():
        click.echo("Empty message. Please provide input or type 'exit' to quit.")
        return

    messages.append({"role": "user", "content": user_input})

    # Indicate that assistant is processing
    click.echo()
    click.secho("Assistant is processing your input. Please wait...", fg="green")

    # Process assistant's response
    final_response = chat_llm.call(
        messages=messages,
        tools=[cropio_tool_schema],
        available_functions=available_functions,
    )

    messages.append({"role": "assistant", "content": final_response})
    click.secho(f"\nAssistant: {final_response}\n", fg="green")


def generate_cropio_tool_schema(cropio_inputs: ChatInputs) -> dict:
    """
    Dynamically build a Littellm 'function' schema for the given cropio.

    cropio_name: The name of the cropio (used for the function 'name').
    cropio_inputs: A ChatInputs object containing cropio_description
                 and a list of input fields (each with a name & description).
    """
    properties = {}
    for field in cropio_inputs.inputs:
        properties[field.name] = {
            "type": "string",
            "description": field.description or "No description provided",
        }

    required_fields = [field.name for field in cropio_inputs.inputs]

    return {
        "type": "function",
        "function": {
            "name": cropio_inputs.cropio_name,
            "description": cropio_inputs.cropio_description or "No cropio description",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
            },
        },
    }


def run_cropio_tool(cropio: Cropio, messages: List[Dict[str, str]], **kwargs):
    """
    Runs the cropio using cropio.takeoff(inputs=kwargs) and returns the output.

    Args:
        cropio (Cropio): The cropio instance to run.
        messages (List[Dict[str, str]]): The chat messages up to this point.
        **kwargs: The inputs collected from the user.

    Returns:
        str: The output from the cropio's execution.

    Raises:
        SystemExit: Exits the chat if an error occurs during cropio execution.
    """
    try:
        # Serialize 'messages' to JSON string before adding to kwargs
        kwargs["cropio_chat_messages"] = json.dumps(messages)

        # Run the cropio with the provided inputs
        cropio_output = cropio.takeoff(inputs=kwargs)

        # Convert CropioOutput to a string to send back to the user
        result = str(cropio_output)

        return result
    except Exception as e:
        # Exit the chat and show the error message
        click.secho("An error occurred while running the cropio:", fg="red")
        click.secho(str(e), fg="red")
        sys.exit(1)


def load_cropio_and_name() -> Tuple[Cropio, str]:
    """
    Loads the cropio by importing the cropio class from the user's project.

    Returns:
        Tuple[Cropio, str]: A tuple containing the Cropio instance and the name of the cropio.
    """
    # Get the current working directory
    cwd = Path.cwd()

    # Path to the pyproject.toml file
    pyproject_path = cwd / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found in the current directory.")

    # Load the pyproject.toml file using 'tomli'
    with pyproject_path.open("rb") as f:
        pyproject_data = tomli.load(f)

    # Get the project name from the 'project' section
    project_name = pyproject_data["project"]["name"]
    folder_name = project_name

    # Derive the cropio class name from the project name
    # E.g., if project_name is 'my_project', cropio_class_name is 'MyProject'
    cropio_class_name = project_name.replace("_", " ").title().replace(" ", "")

    # Add the 'src' directory to sys.path
    src_path = cwd / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Import the cropio module
    cropio_module_name = f"{folder_name}.cropio"
    try:
        cropio_module = __import__(cropio_module_name, fromlist=[cropio_class_name])
    except ImportError as e:
        raise ImportError(f"Failed to import cropio module {cropio_module_name}: {e}")

    # Get the cropio class from the module
    try:
        cropio_class = getattr(cropio_module, cropio_class_name)
    except AttributeError:
        raise AttributeError(
            f"Cropio class {cropio_class_name} not found in module {cropio_module_name}"
        )

    # Instantiate the cropio
    cropio_instance = cropio_class().cropio()
    return cropio_instance, cropio_class_name


def generate_cropio_chat_inputs(cropio: Cropio, cropio_name: str, chat_llm) -> ChatInputs:
    """
    Generates the ChatInputs required for the cropio by analyzing the tasks and agents.

    Args:
        cropio (Cropio): The cropio object containing tasks and agents.
        cropio_name (str): The name of the cropio.
        chat_llm: The chat language model to use for AI calls.

    Returns:
        ChatInputs: An object containing the cropio's name, description, and input fields.
    """
    # Extract placeholders from tasks and agents
    required_inputs = fetch_required_inputs(cropio)

    # Generate descriptions for each input using AI
    input_fields = []
    for input_name in required_inputs:
        description = generate_input_description_with_ai(input_name, cropio, chat_llm)
        input_fields.append(ChatInputField(name=input_name, description=description))

    # Generate cropio description using AI
    cropio_description = generate_cropio_description_with_ai(cropio, chat_llm)

    return ChatInputs(
        cropio_name=cropio_name, cropio_description=cropio_description, inputs=input_fields
    )


def fetch_required_inputs(cropio: Cropio) -> Set[str]:
    """
    Extracts placeholders from the cropio's tasks and agents.

    Args:
        cropio (Cropio): The cropio object.

    Returns:
        Set[str]: A set of placeholder names.
    """
    placeholder_pattern = re.compile(r"\{(.+?)\}")
    required_inputs: Set[str] = set()

    # Scan tasks
    for task in cropio.tasks:
        text = f"{task.description or ''} {task.expected_output or ''}"
        required_inputs.update(placeholder_pattern.findall(text))

    # Scan agents
    for agent in cropio.agents:
        text = f"{agent.role or ''} {agent.goal or ''} {agent.backstory or ''}"
        required_inputs.update(placeholder_pattern.findall(text))

    return required_inputs


def generate_input_description_with_ai(input_name: str, cropio: Cropio, chat_llm) -> str:
    """
    Generates an input description using AI based on the context of the cropio.

    Args:
        input_name (str): The name of the input placeholder.
        cropio (Cropio): The cropio object.
        chat_llm: The chat language model to use for AI calls.

    Returns:
        str: A concise description of the input.
    """
    # Gather context from tasks and agents where the input is used
    context_texts = []
    placeholder_pattern = re.compile(r"\{(.+?)\}")

    for task in cropio.tasks:
        if (
            f"{{{input_name}}}" in task.description
            or f"{{{input_name}}}" in task.expected_output
        ):
            # Replace placeholders with input names
            task_description = placeholder_pattern.sub(
                lambda m: m.group(1), task.description or ""
            )
            expected_output = placeholder_pattern.sub(
                lambda m: m.group(1), task.expected_output or ""
            )
            context_texts.append(f"Task Description: {task_description}")
            context_texts.append(f"Expected Output: {expected_output}")
    for agent in cropio.agents:
        if (
            f"{{{input_name}}}" in agent.role
            or f"{{{input_name}}}" in agent.goal
            or f"{{{input_name}}}" in agent.backstory
        ):
            # Replace placeholders with input names
            agent_role = placeholder_pattern.sub(lambda m: m.group(1), agent.role or "")
            agent_goal = placeholder_pattern.sub(lambda m: m.group(1), agent.goal or "")
            agent_backstory = placeholder_pattern.sub(
                lambda m: m.group(1), agent.backstory or ""
            )
            context_texts.append(f"Agent Role: {agent_role}")
            context_texts.append(f"Agent Goal: {agent_goal}")
            context_texts.append(f"Agent Backstory: {agent_backstory}")

    context = "\n".join(context_texts)
    if not context:
        # If no context is found for the input, raise an exception as per instruction
        raise ValueError(f"No context found for input '{input_name}'.")

    prompt = (
        f"Based on the following context, write a concise description (15 words or less) of the input '{input_name}'.\n"
        "Provide only the description, without any extra text or labels. Do not include placeholders like '{topic}' in the description.\n"
        "Context:\n"
        f"{context}"
    )
    response = chat_llm.call(messages=[{"role": "user", "content": prompt}])
    description = response.strip()

    return description


def generate_cropio_description_with_ai(cropio: Cropio, chat_llm) -> str:
    """
    Generates a brief description of the cropio using AI.

    Args:
        cropio (Cropio): The cropio object.
        chat_llm: The chat language model to use for AI calls.

    Returns:
        str: A concise description of the cropio's purpose (15 words or less).
    """
    # Gather context from tasks and agents
    context_texts = []
    placeholder_pattern = re.compile(r"\{(.+?)\}")

    for task in cropio.tasks:
        # Replace placeholders with input names
        task_description = placeholder_pattern.sub(
            lambda m: m.group(1), task.description or ""
        )
        expected_output = placeholder_pattern.sub(
            lambda m: m.group(1), task.expected_output or ""
        )
        context_texts.append(f"Task Description: {task_description}")
        context_texts.append(f"Expected Output: {expected_output}")
    for agent in cropio.agents:
        # Replace placeholders with input names
        agent_role = placeholder_pattern.sub(lambda m: m.group(1), agent.role or "")
        agent_goal = placeholder_pattern.sub(lambda m: m.group(1), agent.goal or "")
        agent_backstory = placeholder_pattern.sub(
            lambda m: m.group(1), agent.backstory or ""
        )
        context_texts.append(f"Agent Role: {agent_role}")
        context_texts.append(f"Agent Goal: {agent_goal}")
        context_texts.append(f"Agent Backstory: {agent_backstory}")

    context = "\n".join(context_texts)
    if not context:
        raise ValueError("No context found for generating cropio description.")

    prompt = (
        "Based on the following context, write a concise, action-oriented description (15 words or less) of the cropio's purpose.\n"
        "Provide only the description, without any extra text or labels. Do not include placeholders like '{topic}' in the description.\n"
        "Context:\n"
        f"{context}"
    )
    response = chat_llm.call(messages=[{"role": "user", "content": prompt}])
    cropio_description = response.strip()

    return cropio_description
