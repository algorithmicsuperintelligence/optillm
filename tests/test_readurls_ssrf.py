import socket
import unittest
from unittest.mock import ANY, MagicMock, patch

from optillm.plugins.readurls_plugin import fetch_webpage_content, is_safe_url


class TestReadUrlsSSRFProtection(unittest.TestCase):
    @patch("optillm.plugins.readurls_plugin.requests.Session")
    @patch("optillm.plugins.readurls_plugin.socket.getaddrinfo")
    def test_does_not_fetch_loopback_url(self, mock_getaddrinfo, mock_session):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
        ]

        result = fetch_webpage_content("http://localhost:8080/admin")

        self.assertIn("blocked", result.lower())
        mock_session.assert_not_called()

    @patch("optillm.plugins.readurls_plugin.requests.Session")
    @patch("optillm.plugins.readurls_plugin.socket.getaddrinfo")
    def test_blocks_redirect_to_non_public_address(self, mock_getaddrinfo, mock_session):
        mock_getaddrinfo.side_effect = [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
        ]
        redirect_response = MagicMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {"Location": "http://localhost/admin"}
        mock_session.return_value.get.return_value = redirect_response

        result = fetch_webpage_content("https://example.com")

        self.assertIn("blocked", result.lower())
        mock_session.return_value.get.assert_called_once_with(
            "https://example.com", headers=ANY, timeout=10,
            verify=True, allow_redirects=False
        )

    @patch("optillm.plugins.readurls_plugin.socket.getaddrinfo")
    def test_rejects_host_with_any_non_public_address(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0)),
        ]

        self.assertFalse(is_safe_url("https://example.com"))

    @patch("optillm.plugins.readurls_plugin.socket.getaddrinfo")
    def test_allows_host_with_only_public_addresses(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ]

        self.assertTrue(is_safe_url("https://example.com"))


if __name__ == "__main__":
    unittest.main()
