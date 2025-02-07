import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from requests.exceptions import JSONDecodeError

from cropioai.cli.deploy.main import DeployCommand
from cropioai.cli.utils import parse_toml


class TestDeployCommand(unittest.TestCase):
    @patch("cropioai.cli.command.get_auth_token")
    @patch("cropioai.cli.deploy.main.get_project_name")
    @patch("cropioai.cli.command.PlusAPI")
    def setUp(self, mock_plus_api, mock_get_project_name, mock_get_auth_token):
        self.mock_get_auth_token = mock_get_auth_token
        self.mock_get_project_name = mock_get_project_name
        self.mock_plus_api = mock_plus_api

        self.mock_get_auth_token.return_value = "test_token"
        self.mock_get_project_name.return_value = "test_project"

        self.deploy_command = DeployCommand()
        self.mock_client = self.deploy_command.plus_api_client

    def test_init_success(self):
        self.assertEqual(self.deploy_command.project_name, "test_project")
        self.mock_plus_api.assert_called_once_with(api_key="test_token")

    @patch("cropioai.cli.command.get_auth_token")
    def test_init_failure(self, mock_get_auth_token):
        mock_get_auth_token.side_effect = Exception("Auth failed")

        with self.assertRaises(SystemExit):
            DeployCommand()

    def test_validate_response_successful_response(self):
        mock_response = Mock(spec=requests.Response)
        mock_response.json.return_value = {"message": "Success"}
        mock_response.status_code = 200
        mock_response.ok = True

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._validate_response(mock_response)
            assert fake_out.getvalue() == ""

    def test_validate_response_json_decode_error(self):
        mock_response = Mock(spec=requests.Response)
        mock_response.json.side_effect = JSONDecodeError("Decode error", "", 0)
        mock_response.status_code = 500
        mock_response.content = b"Invalid JSON"

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit):
                self.deploy_command._validate_response(mock_response)
            output = fake_out.getvalue()
            assert (
                "Failed to parse response from Enterprise API failed. Details:"
                in output
            )
            assert "Status Code: 500" in output
            assert "Response:\nb'Invalid JSON'" in output

    def test_validate_response_422_error(self):
        mock_response = Mock(spec=requests.Response)
        mock_response.json.return_value = {
            "field1": ["Error message 1"],
            "field2": ["Error message 2"],
        }
        mock_response.status_code = 422
        mock_response.ok = False

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit):
                self.deploy_command._validate_response(mock_response)
            output = fake_out.getvalue()
            assert (
                "Failed to complete operation. Please fix the following errors:"
                in output
            )
            assert "Field1 Error message 1" in output
            assert "Field2 Error message 2" in output

    def test_validate_response_other_error(self):
        mock_response = Mock(spec=requests.Response)
        mock_response.json.return_value = {"error": "Something went wrong"}
        mock_response.status_code = 500
        mock_response.ok = False

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit):
                self.deploy_command._validate_response(mock_response)
            output = fake_out.getvalue()
            assert "Request to Enterprise API failed. Details:" in output
            assert "Details:\nSomething went wrong" in output

    def test_standard_no_param_error_message(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._standard_no_param_error_message()
            self.assertIn("No UUID provided", fake_out.getvalue())

    def test_display_deployment_info(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_deployment_info(
                {"uuid": "test-uuid", "status": "deployed"}
            )
            self.assertIn("Deploying the cropio...", fake_out.getvalue())
            self.assertIn("test-uuid", fake_out.getvalue())
            self.assertIn("deployed", fake_out.getvalue())

    def test_display_logs(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_logs(
                [{"timestamp": "2023-01-01", "level": "INFO", "message": "Test log"}]
            )
            self.assertIn("2023-01-01 - INFO: Test log", fake_out.getvalue())

    @patch("cropioai.cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_uuid(self, mock_display):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_uuid.return_value = mock_response

        self.deploy_command.deploy(uuid="test-uuid")

        self.mock_client.deploy_by_uuid.assert_called_once_with("test-uuid")
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("cropioai.cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_project_name(self, mock_display):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_name.return_value = mock_response

        self.deploy_command.deploy()

        self.mock_client.deploy_by_name.assert_called_once_with("test_project")
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("cropioai.cli.deploy.main.fetch_and_json_env_file")
    @patch("cropioai.cli.deploy.main.git.Repository.origin_url")
    @patch("builtins.input")
    def test_create_cropio(self, mock_input, mock_git_origin_url, mock_fetch_env):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_git_origin_url.return_value = "https://github.com/test/repo.git"
        mock_input.return_value = ""

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"uuid": "new-uuid", "status": "created"}
        self.mock_client.create_cropio.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.create_cropio()
            self.assertIn("Deployment created successfully!", fake_out.getvalue())
            self.assertIn("new-uuid", fake_out.getvalue())

    def test_list_cropios(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "Cropio1", "uuid": "uuid1", "status": "active"},
            {"name": "Cropio2", "uuid": "uuid2", "status": "inactive"},
        ]
        self.mock_client.list_cropios.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.list_cropios()
            self.assertIn("Cropio1 (uuid1) active", fake_out.getvalue())
            self.assertIn("Cropio2 (uuid2) inactive", fake_out.getvalue())

    def test_get_cropio_status(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "InternalCropio", "status": "active"}
        self.mock_client.cropio_status_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.get_cropio_status()
            self.assertIn("InternalCropio", fake_out.getvalue())
            self.assertIn("active", fake_out.getvalue())

    def test_get_cropio_logs(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"timestamp": "2023-01-01", "level": "INFO", "message": "Log1"},
            {"timestamp": "2023-01-02", "level": "ERROR", "message": "Log2"},
        ]
        self.mock_client.cropio_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.get_cropio_logs(None)
            self.assertIn("2023-01-01 - INFO: Log1", fake_out.getvalue())
            self.assertIn("2023-01-02 - ERROR: Log2", fake_out.getvalue())

    def test_remove_cropio(self):
        mock_response = MagicMock()
        mock_response.status_code = 204
        self.mock_client.delete_cropio_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.remove_cropio(None)
            self.assertIn(
                "Cropio 'test_project' removed successfully", fake_out.getvalue()
            )

    @unittest.skipIf(sys.version_info < (3, 11), "Requires Python 3.11+")
    def test_parse_toml_python_311_plus(self):
        toml_content = """
        [tool.poetry]
        name = "test_project"
        version = "0.4.0"

        [tool.poetry.dependencies]
        python = "^3.11"
        cropioai = { extras = ["tools"], version = ">=0.51.0,<1.0.0" }
        """
        parsed = parse_toml(toml_content)
        self.assertEqual(parsed["tool"]["poetry"]["name"], "test_project")

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="""
        [project]
        name = "test_project"
        version = "0.4.0"
        requires-python = ">=3.10,<3.13"
        dependencies = ["cropioai"]
        """,
    )
    def test_get_project_name_python_310(self, mock_open):
        from cropioai.cli.utils import get_project_name

        project_name = get_project_name()
        print("project_name", project_name)
        self.assertEqual(project_name, "test_project")

    @unittest.skipIf(sys.version_info < (3, 11), "Requires Python 3.11+")
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="""
    [project]
    name = "test_project"
    version = "0.4.0"
    requires-python = ">=3.10,<3.13"
    dependencies = ["cropioai"]
    """,
    )
    def test_get_project_name_python_311_plus(self, mock_open):
        from cropioai.cli.utils import get_project_name

        project_name = get_project_name()
        self.assertEqual(project_name, "test_project")

    def test_get_cropioai_version(self):
        from cropioai.cli.version import get_cropioai_version

        assert isinstance(get_cropioai_version(), str)
