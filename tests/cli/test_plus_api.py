import os
import unittest
from unittest.mock import MagicMock, patch

from cropioai.cli.plus_api import PlusAPI


class TestPlusAPI(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.api = PlusAPI(self.api_key)

    def test_init(self):
        self.assertEqual(self.api.api_key, self.api_key)
        self.assertEqual(self.api.headers["Authorization"], f"Bearer {self.api_key}")
        self.assertEqual(self.api.headers["Content-Type"], "application/json")
        self.assertTrue("CropioAI-CLI/" in self.api.headers["User-Agent"])
        self.assertTrue(self.api.headers["X-Cropioai-Version"])

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_login_to_tool_repository(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.login_to_tool_repository()

        mock_make_request.assert_called_once_with(
            "POST", "/cropioai_plus/api/v1/tools/login"
        )
        self.assertEqual(response, mock_response)

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_get_tool(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.get_tool("test_tool_handle")

        mock_make_request.assert_called_once_with(
            "GET", "/cropioai_plus/api/v1/tools/test_tool_handle"
        )
        self.assertEqual(response, mock_response)

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_publish_tool(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response
        handle = "test_tool_handle"
        public = True
        version = "1.0.0"
        description = "Test tool description"
        encoded_file = "encoded_test_file"

        response = self.api.publish_tool(
            handle, public, version, description, encoded_file
        )

        params = {
            "handle": handle,
            "public": public,
            "version": version,
            "file": encoded_file,
            "description": description,
        }
        mock_make_request.assert_called_once_with(
            "POST", "/cropioai_plus/api/v1/tools", json=params
        )
        self.assertEqual(response, mock_response)

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_publish_tool_without_description(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response
        handle = "test_tool_handle"
        public = False
        version = "2.0.0"
        description = None
        encoded_file = "encoded_test_file"

        response = self.api.publish_tool(
            handle, public, version, description, encoded_file
        )

        params = {
            "handle": handle,
            "public": public,
            "version": version,
            "file": encoded_file,
            "description": description,
        }
        mock_make_request.assert_called_once_with(
            "POST", "/cropioai_plus/api/v1/tools", json=params
        )
        self.assertEqual(response, mock_response)

    @patch("cropioai.cli.plus_api.requests.Session")
    def test_make_request(self, mock_session):
        mock_response = MagicMock()

        mock_session_instance = mock_session.return_value
        mock_session_instance.request.return_value = mock_response

        response = self.api._make_request("GET", "test_endpoint")

        mock_session.assert_called_once()
        mock_session_instance.request.assert_called_once_with(
            "GET", f"{self.api.base_url}/test_endpoint", headers=self.api.headers
        )
        mock_session_instance.trust_env = False
        self.assertEqual(response, mock_response)

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_deploy_by_name(self, mock_make_request):
        self.api.deploy_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "POST", "/cropioai_plus/api/v1/cropios/by-name/test_project/deploy"
        )

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_deploy_by_uuid(self, mock_make_request):
        self.api.deploy_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with(
            "POST", "/cropioai_plus/api/v1/cropios/test_uuid/deploy"
        )

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_cropio_status_by_name(self, mock_make_request):
        self.api.cropio_status_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "GET", "/cropioai_plus/api/v1/cropios/by-name/test_project/status"
        )

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_cropio_status_by_uuid(self, mock_make_request):
        self.api.cropio_status_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with(
            "GET", "/cropioai_plus/api/v1/cropios/test_uuid/status"
        )

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_cropio_by_name(self, mock_make_request):
        self.api.cropio_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "GET", "/cropioai_plus/api/v1/cropios/by-name/test_project/logs/deployment"
        )

        self.api.cropio_by_name("test_project", "custom_log")
        mock_make_request.assert_called_with(
            "GET", "/cropioai_plus/api/v1/cropios/by-name/test_project/logs/custom_log"
        )

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_cropio_by_uuid(self, mock_make_request):
        self.api.cropio_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with(
            "GET", "/cropioai_plus/api/v1/cropios/test_uuid/logs/deployment"
        )

        self.api.cropio_by_uuid("test_uuid", "custom_log")
        mock_make_request.assert_called_with(
            "GET", "/cropioai_plus/api/v1/cropios/test_uuid/logs/custom_log"
        )

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_delete_cropio_by_name(self, mock_make_request):
        self.api.delete_cropio_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "DELETE", "/cropioai_plus/api/v1/cropios/by-name/test_project"
        )

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_delete_cropio_by_uuid(self, mock_make_request):
        self.api.delete_cropio_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with(
            "DELETE", "/cropioai_plus/api/v1/cropios/test_uuid"
        )

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_list_cropios(self, mock_make_request):
        self.api.list_cropios()
        mock_make_request.assert_called_once_with("GET", "/cropioai_plus/api/v1/cropios")

    @patch("cropioai.cli.plus_api.PlusAPI._make_request")
    def test_create_cropio(self, mock_make_request):
        payload = {"name": "test_cropio"}
        self.api.create_cropio(payload)
        mock_make_request.assert_called_once_with(
            "POST", "/cropioai_plus/api/v1/cropios", json=payload
        )

    @patch.dict(os.environ, {"CREWAI_BASE_URL": "https://custom-url.com/api"})
    def test_custom_base_url(self):
        custom_api = PlusAPI("test_key")
        self.assertEqual(
            custom_api.base_url,
            "https://custom-url.com/api",
        )
