"""
Unit tests for SSL configuration support in optillm.

Tests verify that SSL certificate verification can be configured via:
- Command-line arguments (--ssl-verify, --no-ssl-verify, --ssl-cert-path)
- Environment variables (OPTILLM_SSL_VERIFY, OPTILLM_SSL_CERT_PATH)
- And that SSL settings are properly propagated to HTTP clients
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import optillm modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm import server_config, parse_args


class TestSSLConfiguration(unittest.TestCase):
    """Test SSL configuration via CLI arguments and environment variables."""

    def setUp(self):
        """Reset server_config before each test."""
        # Save original config
        self.original_config = server_config.copy()

        # Clear SSL-related environment variables
        for key in ['OPTILLM_SSL_VERIFY', 'OPTILLM_SSL_CERT_PATH']:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        """Restore original server_config after each test."""
        server_config.clear()
        server_config.update(self.original_config)

    def test_default_ssl_verify_enabled(self):
        """Test that SSL verification is enabled by default."""
        self.assertTrue(server_config.get('ssl_verify', True))
        self.assertEqual(server_config.get('ssl_cert_path', ''), '')

    def test_cli_no_ssl_verify_flag(self):
        """Test --no-ssl-verify CLI flag disables SSL verification."""
        with patch('sys.argv', ['optillm', '--no-ssl-verify']):
            args = parse_args()
            self.assertFalse(args.ssl_verify)

    def test_cli_ssl_cert_path(self):
        """Test --ssl-cert-path CLI argument."""
        test_cert_path = '/path/to/ca-bundle.crt'
        with patch('sys.argv', ['optillm', '--ssl-cert-path', test_cert_path]):
            args = parse_args()
            self.assertEqual(args.ssl_cert_path, test_cert_path)

    def test_env_ssl_verify_false(self):
        """Test OPTILLM_SSL_VERIFY=false environment variable."""
        os.environ['OPTILLM_SSL_VERIFY'] = 'false'
        with patch('sys.argv', ['optillm']):
            args = parse_args()
            self.assertFalse(args.ssl_verify)

    def test_env_ssl_verify_true(self):
        """Test OPTILLM_SSL_VERIFY=true environment variable."""
        os.environ['OPTILLM_SSL_VERIFY'] = 'true'
        with patch('sys.argv', ['optillm']):
            args = parse_args()
            self.assertTrue(args.ssl_verify)

    def test_env_ssl_cert_path(self):
        """Test OPTILLM_SSL_CERT_PATH environment variable."""
        test_cert_path = '/etc/ssl/certs/custom-ca.pem'
        os.environ['OPTILLM_SSL_CERT_PATH'] = test_cert_path
        with patch('sys.argv', ['optillm']):
            args = parse_args()
            self.assertEqual(args.ssl_cert_path, test_cert_path)

    def test_cli_overrides_env(self):
        """Test that CLI arguments override environment variables."""
        os.environ['OPTILLM_SSL_VERIFY'] = 'true'
        with patch('sys.argv', ['optillm', '--no-ssl-verify']):
            args = parse_args()
            self.assertFalse(args.ssl_verify)


class TestHTTPClientSSLConfiguration(unittest.TestCase):
    """Test that SSL configuration is properly applied to HTTP clients."""

    def setUp(self):
        """Set up test environment."""
        self.original_config = server_config.copy()

    def tearDown(self):
        """Restore original server_config."""
        server_config.clear()
        server_config.update(self.original_config)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_httpx_client_ssl_verify_disabled(self):
        """Test httpx.Client created with verify=False when SSL disabled."""
        from optillm.server import get_config

        # Configure to disable SSL verification
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''

        # Create client
        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI'):
            get_config()
            # Verify httpx.Client was called with verify=False
            mock_httpx_client.assert_called_once_with(verify=False)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_httpx_client_ssl_verify_enabled(self):
        """Test httpx.Client created with verify=True by default."""
        from optillm.server import get_config

        # Configure to enable SSL verification (default)
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''

        # Create client
        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI'):
            get_config()
            # Verify httpx.Client was called with verify=True
            mock_httpx_client.assert_called_once_with(verify=True)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_httpx_client_custom_cert_path(self):
        """Test httpx.Client created with custom certificate path."""
        from optillm.server import get_config

        # Configure custom certificate path
        test_cert_path = '/path/to/custom-ca.pem'
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = test_cert_path

        # Create client
        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI'):
            get_config()
            # Verify httpx.Client was called with custom cert path
            mock_httpx_client.assert_called_once_with(verify=test_cert_path)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'OPTILLM_API_KEY': ''}, clear=False)
    def test_openai_client_receives_http_client(self):
        """Test that OpenAI client receives the configured httpx client."""
        from optillm.server import get_config

        # Ensure OPTILLM_API_KEY is not set (it takes precedence)
        if 'OPTILLM_API_KEY' in os.environ:
            del os.environ['OPTILLM_API_KEY']

        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        mock_http_client_instance = MagicMock()

        with patch('httpx.Client', return_value=mock_http_client_instance), \
             patch('optillm.server.OpenAI') as mock_openai:
            get_config()

            # Verify OpenAI was called with http_client parameter
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            self.assertIn('http_client', call_kwargs)
            self.assertEqual(call_kwargs['http_client'], mock_http_client_instance)

    @patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test-key', 'OPTILLM_API_KEY': ''}, clear=False)
    def test_cerebras_client_receives_http_client(self):
        """Test that Cerebras client receives the configured httpx client."""
        from optillm.server import get_config

        # Ensure OPTILLM_API_KEY is not set (it takes precedence)
        if 'OPTILLM_API_KEY' in os.environ:
            del os.environ['OPTILLM_API_KEY']

        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        mock_http_client_instance = MagicMock()

        with patch('httpx.Client', return_value=mock_http_client_instance), \
             patch('optillm.server.Cerebras') as mock_cerebras:
            get_config()

            # Verify Cerebras was called with http_client parameter
            mock_cerebras.assert_called_once()
            call_kwargs = mock_cerebras.call_args[1]
            self.assertIn('http_client', call_kwargs)
            self.assertEqual(call_kwargs['http_client'], mock_http_client_instance)

    @patch.dict(os.environ, {'AZURE_OPENAI_API_KEY': 'test-key', 'AZURE_API_VERSION': '2024-02-15-preview', 'AZURE_API_BASE': 'https://test.openai.azure.com', 'OPTILLM_API_KEY': ''}, clear=False)
    def test_azure_client_receives_http_client(self):
        """Test that AzureOpenAI client receives the configured httpx client."""
        from optillm.server import get_config

        # Ensure OPTILLM_API_KEY is not set (it takes precedence)
        if 'OPTILLM_API_KEY' in os.environ:
            del os.environ['OPTILLM_API_KEY']

        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''

        mock_http_client_instance = MagicMock()

        with patch('httpx.Client', return_value=mock_http_client_instance), \
             patch('optillm.server.AzureOpenAI') as mock_azure:
            get_config()

            # Verify AzureOpenAI was called with http_client parameter
            mock_azure.assert_called_once()
            call_kwargs = mock_azure.call_args[1]
            self.assertIn('http_client', call_kwargs)
            self.assertEqual(call_kwargs['http_client'], mock_http_client_instance)


class TestPluginSSLConfiguration(unittest.TestCase):
    """Test that plugins properly use SSL configuration."""

    def setUp(self):
        """Set up test environment."""
        self.original_config = server_config.copy()

    def tearDown(self):
        """Restore original server_config."""
        server_config.clear()
        server_config.update(self.original_config)

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_readurls_plugin_ssl_verify_disabled(self, mock_requests_get):
        """Test readurls plugin respects SSL verification disabled."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content

        # Configure to disable SSL verification
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''

        # Mock response
        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Test content</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        # Fetch webpage
        fetch_webpage_content('https://example.com')

        # Verify requests.get was called with verify=False
        mock_requests_get.assert_called_once()
        call_kwargs = mock_requests_get.call_args[1]
        self.assertIn('verify', call_kwargs)
        self.assertFalse(call_kwargs['verify'])

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_readurls_plugin_ssl_verify_enabled(self, mock_requests_get):
        """Test readurls plugin respects SSL verification enabled."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content

        # Configure to enable SSL verification
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''

        # Mock response
        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Test content</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        # Fetch webpage
        fetch_webpage_content('https://example.com')

        # Verify requests.get was called with verify=True
        mock_requests_get.assert_called_once()
        call_kwargs = mock_requests_get.call_args[1]
        self.assertIn('verify', call_kwargs)
        self.assertTrue(call_kwargs['verify'])

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_readurls_plugin_custom_cert_path(self, mock_requests_get):
        """Test readurls plugin uses custom certificate path."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content

        # Configure custom certificate path
        test_cert_path = '/path/to/custom-ca.pem'
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = test_cert_path

        # Mock response
        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Test content</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        # Fetch webpage
        fetch_webpage_content('https://example.com')

        # Verify requests.get was called with custom cert path
        mock_requests_get.assert_called_once()
        call_kwargs = mock_requests_get.call_args[1]
        self.assertIn('verify', call_kwargs)
        self.assertEqual(call_kwargs['verify'], test_cert_path)


class TestSSLWarnings(unittest.TestCase):
    """Test that appropriate warnings are shown when SSL is disabled."""

    def setUp(self):
        """Set up test environment."""
        self.original_config = server_config.copy()

    def tearDown(self):
        """Restore original server_config."""
        server_config.clear()
        server_config.update(self.original_config)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_warning_when_ssl_disabled(self):
        """Test that a warning is logged when SSL verification is disabled."""
        from optillm.server import get_config

        # Configure to disable SSL verification
        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''

        with patch('httpx.Client'), \
             patch('optillm.server.OpenAI'), \
             patch('optillm.server.logger.warning') as mock_logger_warning:
            get_config()

            # Verify warning was logged
            mock_logger_warning.assert_called()
            warning_message = mock_logger_warning.call_args[0][0]
            self.assertIn('SSL certificate verification is DISABLED', warning_message)
            self.assertIn('insecure', warning_message.lower())

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'OPTILLM_API_KEY': ''}, clear=False)
    def test_info_when_custom_cert_used(self):
        """Test that an info message is logged when using custom certificate."""
        from optillm.server import get_config

        # Ensure OPTILLM_API_KEY is not set (it takes precedence)
        if 'OPTILLM_API_KEY' in os.environ:
            del os.environ['OPTILLM_API_KEY']

        # Configure custom certificate path
        test_cert_path = '/path/to/custom-ca.pem'
        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = test_cert_path

        with patch('httpx.Client'), \
             patch('optillm.server.OpenAI'), \
             patch('optillm.server.logger.info') as mock_logger_info:
            get_config()

            # Verify info message was logged about custom cert
            # The logger.info is called multiple times, check all calls
            all_info_messages = [call[0][0] for call in mock_logger_info.call_args_list if call[0]]
            cert_message_found = any('custom CA certificate bundle' in msg for msg in all_info_messages)
            self.assertTrue(cert_message_found, f"Expected 'custom CA certificate bundle' in one of: {all_info_messages}")


if __name__ == '__main__':
    unittest.main()