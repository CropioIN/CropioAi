from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from cropioai.cli.cli import (
    deploy_create,
    deploy_list,
    deploy_logs,
    deploy_push,
    deploy_remove,
    deply_status,
    flow_add_cropio,
    reset_memories,
    signup,
    test,
    train,
    version,
)


@pytest.fixture
def runner():
    return CliRunner()


@mock.patch("cropioai.cli.cli.train_cropio")
def test_train_default_iterations(train_cropio, runner):
    result = runner.invoke(train)

    train_cropio.assert_called_once_with(5, "trained_agents_data.pkl")
    assert result.exit_code == 0
    assert "Training the Cropio for 5 iterations" in result.output


@mock.patch("cropioai.cli.cli.train_cropio")
def test_train_custom_iterations(train_cropio, runner):
    result = runner.invoke(train, ["--n_iterations", "10"])

    train_cropio.assert_called_once_with(10, "trained_agents_data.pkl")
    assert result.exit_code == 0
    assert "Training the Cropio for 10 iterations" in result.output


@mock.patch("cropioai.cli.cli.train_cropio")
def test_train_invalid_string_iterations(train_cropio, runner):
    result = runner.invoke(train, ["--n_iterations", "invalid"])

    train_cropio.assert_not_called()
    assert result.exit_code == 2
    assert (
        "Usage: train [OPTIONS]\nTry 'train --help' for help.\n\nError: Invalid value for '-n' / '--n_iterations': 'invalid' is not a valid integer.\n"
        in result.output
    )


@mock.patch("cropioai.cli.reset_memories_command.ShortTermMemory")
@mock.patch("cropioai.cli.reset_memories_command.EntityMemory")
@mock.patch("cropioai.cli.reset_memories_command.LongTermMemory")
@mock.patch("cropioai.cli.reset_memories_command.TaskOutputStorageHandler")
def test_reset_all_memories(
    MockTaskOutputStorageHandler,
    MockLongTermMemory,
    MockEntityMemory,
    MockShortTermMemory,
    runner,
):
    result = runner.invoke(reset_memories, ["--all"])
    MockShortTermMemory().reset.assert_called_once()
    MockEntityMemory().reset.assert_called_once()
    MockLongTermMemory().reset.assert_called_once()
    MockTaskOutputStorageHandler().reset.assert_called_once()

    assert result.output == "All memories have been reset.\n"


@mock.patch("cropioai.cli.reset_memories_command.ShortTermMemory")
def test_reset_short_term_memories(MockShortTermMemory, runner):
    result = runner.invoke(reset_memories, ["-s"])
    MockShortTermMemory().reset.assert_called_once()
    assert result.output == "Short term memory has been reset.\n"


@mock.patch("cropioai.cli.reset_memories_command.EntityMemory")
def test_reset_entity_memories(MockEntityMemory, runner):
    result = runner.invoke(reset_memories, ["-e"])
    MockEntityMemory().reset.assert_called_once()
    assert result.output == "Entity memory has been reset.\n"


@mock.patch("cropioai.cli.reset_memories_command.LongTermMemory")
def test_reset_long_term_memories(MockLongTermMemory, runner):
    result = runner.invoke(reset_memories, ["-l"])
    MockLongTermMemory().reset.assert_called_once()
    assert result.output == "Long term memory has been reset.\n"


@mock.patch("cropioai.cli.reset_memories_command.TaskOutputStorageHandler")
def test_reset_ignite_outputs(MockTaskOutputStorageHandler, runner):
    result = runner.invoke(reset_memories, ["-k"])
    MockTaskOutputStorageHandler().reset.assert_called_once()
    assert result.output == "Latest Ignite outputs stored has been reset.\n"


@mock.patch("cropioai.cli.reset_memories_command.ShortTermMemory")
@mock.patch("cropioai.cli.reset_memories_command.LongTermMemory")
def test_reset_multiple_memory_flags(MockShortTermMemory, MockLongTermMemory, runner):
    result = runner.invoke(
        reset_memories,
        [
            "-s",
            "-l",
        ],
    )
    MockShortTermMemory().reset.assert_called_once()
    MockLongTermMemory().reset.assert_called_once()
    assert (
        result.output
        == "Long term memory has been reset.\nShort term memory has been reset.\n"
    )


def test_reset_no_memory_flags(runner):
    result = runner.invoke(
        reset_memories,
    )
    assert (
        result.output
        == "Please specify at least one memory type to reset using the appropriate flags.\n"
    )


def test_version_flag(runner):
    result = runner.invoke(version)

    assert result.exit_code == 0
    assert "cropioai version:" in result.output


def test_version_command(runner):
    result = runner.invoke(version)

    assert result.exit_code == 0
    assert "cropioai version:" in result.output


def test_version_command_with_tools(runner):
    result = runner.invoke(version, ["--tools"])

    assert result.exit_code == 0
    assert "cropioai version:" in result.output
    assert (
        "cropioai tools version:" in result.output
        or "cropioai tools not installed" in result.output
    )


@mock.patch("cropioai.cli.cli.evaluate_cropio")
def test_test_default_iterations(evaluate_cropio, runner):
    result = runner.invoke(test)

    evaluate_cropio.assert_called_once_with(3, "gpt-4o-mini")
    assert result.exit_code == 0
    assert "Testing the cropio for 3 iterations with model gpt-4o-mini" in result.output


@mock.patch("cropioai.cli.cli.evaluate_cropio")
def test_test_custom_iterations(evaluate_cropio, runner):
    result = runner.invoke(test, ["--n_iterations", "5", "--model", "gpt-4o"])

    evaluate_cropio.assert_called_once_with(5, "gpt-4o")
    assert result.exit_code == 0
    assert "Testing the cropio for 5 iterations with model gpt-4o" in result.output


@mock.patch("cropioai.cli.cli.evaluate_cropio")
def test_test_invalid_string_iterations(evaluate_cropio, runner):
    result = runner.invoke(test, ["--n_iterations", "invalid"])

    evaluate_cropio.assert_not_called()
    assert result.exit_code == 2
    assert (
        "Usage: test [OPTIONS]\nTry 'test --help' for help.\n\nError: Invalid value for '-n' / '--n_iterations': 'invalid' is not a valid integer.\n"
        in result.output
    )


@mock.patch("cropioai.cli.cli.AuthenticationCommand")
def test_signup(command, runner):
    mock_auth = command.return_value
    result = runner.invoke(signup)

    assert result.exit_code == 0
    mock_auth.signup.assert_called_once()


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_create(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_create)

    assert result.exit_code == 0
    mock_deploy.create_cropio.assert_called_once()


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_list(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_list)

    assert result.exit_code == 0
    mock_deploy.list_cropios.assert_called_once()


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_push(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_push, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.deploy.assert_called_once_with(uuid=uuid)


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_push_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_push)

    assert result.exit_code == 0
    mock_deploy.deploy.assert_called_once_with(uuid=None)


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_status(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deply_status, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.get_cropio_status.assert_called_once_with(uuid=uuid)


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_status_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deply_status)

    assert result.exit_code == 0
    mock_deploy.get_cropio_status.assert_called_once_with(uuid=None)


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_logs(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_logs, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.get_cropio_logs.assert_called_once_with(uuid=uuid)


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_logs_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_logs)

    assert result.exit_code == 0
    mock_deploy.get_cropio_logs.assert_called_once_with(uuid=None)


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_remove(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_remove, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.remove_cropio.assert_called_once_with(uuid=uuid)


@mock.patch("cropioai.cli.cli.DeployCommand")
def test_deploy_remove_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_remove)

    assert result.exit_code == 0
    mock_deploy.remove_cropio.assert_called_once_with(uuid=None)


@mock.patch("cropioai.cli.add_cropio_to_flow.create_embedded_cropio")
@mock.patch("pathlib.Path.exists", return_value=True)  # Mock the existence check
def test_flow_add_cropio(mock_path_exists, mock_create_embedded_cropio, runner):
    cropio_name = "new_cropio"
    result = runner.invoke(flow_add_cropio, [cropio_name])

    # Log the output for debugging
    print(result.output)

    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    assert f"Adding cropio {cropio_name} to the flow" in result.output

    # Verify that create_embedded_cropio was called with the correct arguments
    mock_create_embedded_cropio.assert_called_once()
    call_args, call_kwargs = mock_create_embedded_cropio.call_args
    assert call_args[0] == cropio_name
    assert "parent_folder" in call_kwargs
    assert isinstance(call_kwargs["parent_folder"], Path)


def test_add_cropio_to_flow_not_in_root(runner):
    # Simulate not being in the root of a flow project
    with mock.patch("pathlib.Path.exists", autospec=True) as mock_exists:
        # Mock Path.exists to return False when checking for pyproject.toml
        def exists_side_effect(self):
            if self.name == "pyproject.toml":
                return False  # Simulate that pyproject.toml does not exist
            return True  # All other paths exist

        mock_exists.side_effect = exists_side_effect

        result = runner.invoke(flow_add_cropio, ["new_cropio"])

        assert result.exit_code != 0
        assert "This command must be run from the root of a flow project." in str(
            result.output
        )
