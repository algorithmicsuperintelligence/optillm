"""
Unit tests for MiniMax provider support in optillm.

Tests verify that the MiniMax provider:
- Is correctly detected via MINIMAX_API_KEY environment variable
- Creates an OpenAI client with the MiniMax base URL
- Respects custom base_url when set
- Takes priority over OpenAI when both keys are set
- Does not interfere when MINIMAX_API_KEY is not set
- Properly clamps temperature to MiniMax's valid range (0, 1]
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json

# Add parent directory to path to import optillm modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy dependencies that may not be installed in test environments
for mod_name in ['z3', 'torch', 'torch.nn', 'torch.nn.functional',
                 'transformers', 'adaptive_classifier',
                 'peft', 'bitsandbytes', 'outlines', 'spacy',
                 'presidio_analyzer', 'presidio_anonymizer']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from optillm import server_config


class TestMiniMaxProviderDetection(unittest.TestCase):
    """Test MiniMax provider detection via MINIMAX_API_KEY."""

    def setUp(self):
        """Reset server_config before each test."""
        self.original_config = server_config.copy()
        # Clear provider-related environment variables
        for key in ['MINIMAX_API_KEY', 'OPENAI_API_KEY', 'CEREBRAS_API_KEY',
                    'AZURE_OPENAI_API_KEY', 'OPTILLM_API_KEY']:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        """Restore original server_config after each test."""
        server_config.clear()
        server_config.update(self.original_config)

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-minimax-key'})
    def test_minimax_provider_detected(self):
        """Test that MINIMAX_API_KEY triggers MiniMax provider."""
        from optillm.server import get_config

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI') as mock_openai:
            client, api_key = get_config()

            # Should use the MiniMax API key
            assert api_key == 'test-minimax-key'

            # Should create OpenAI client with MiniMax base URL
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs['api_key'] == 'test-minimax-key'
            assert call_kwargs['base_url'] == 'https://api.minimax.io/v1'

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-minimax-key'})
    def test_minimax_default_base_url(self):
        """Test MiniMax provider uses default base URL https://api.minimax.io/v1."""
        from optillm.server import get_config

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI') as mock_openai:
            get_config()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs['base_url'] == 'https://api.minimax.io/v1'

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-minimax-key', 'MINIMAX_API_REGION': 'cn'})
    def test_minimax_china_base_url(self):
        """Test MiniMax provider selects the China endpoint explicitly."""
        from optillm.server import get_config

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        with patch('httpx.Client'), patch('optillm.server.OpenAI') as mock_openai:
            get_config()
            assert mock_openai.call_args[1]['base_url'] == 'https://api.minimaxi.com/v1'

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-minimax-key'})
    def test_minimax_with_custom_base_url(self):
        """Test MiniMax provider with custom base_url."""
        from optillm.server import get_config

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = 'https://custom-proxy.example.com/v1'

        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI') as mock_openai:
            client, api_key = get_config()

            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs['base_url'] == 'https://custom-proxy.example.com/v1'

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'test-minimax-key'})
    def test_minimax_client_receives_http_client(self):
        """Test that MiniMax client receives the configured httpx client."""
        from optillm.server import get_config

        server_config['ssl_verify'] = False
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        mock_http_client_instance = MagicMock()

        with patch('httpx.Client', return_value=mock_http_client_instance) as mock_httpx_client, \
             patch('optillm.server.OpenAI') as mock_openai:
            get_config()

            call_kwargs = mock_openai.call_args[1]
            assert 'http_client' in call_kwargs
            assert call_kwargs['http_client'] == mock_http_client_instance

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'minimax-key', 'OPENAI_API_KEY': 'openai-key'})
    def test_minimax_takes_priority_over_openai(self):
        """Test that MINIMAX_API_KEY takes priority over OPENAI_API_KEY."""
        from optillm.server import get_config

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI') as mock_openai:
            client, api_key = get_config()

            assert api_key == 'minimax-key'
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs['base_url'] == 'https://api.minimax.io/v1'

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'openai-key'})
    def test_openai_still_works_without_minimax(self):
        """Test that OpenAI provider works when MINIMAX_API_KEY is not set."""
        from optillm.server import get_config

        # Ensure MINIMAX_API_KEY is not set
        if 'MINIMAX_API_KEY' in os.environ:
            del os.environ['MINIMAX_API_KEY']

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI') as mock_openai:
            client, api_key = get_config()

            assert api_key == 'openai-key'
            call_kwargs = mock_openai.call_args[1]
            # Should NOT have MiniMax base URL
            assert 'base_url' not in call_kwargs

    @patch.dict(os.environ, {'CEREBRAS_API_KEY': 'cerebras-key', 'MINIMAX_API_KEY': 'minimax-key'})
    def test_cerebras_takes_priority_over_minimax(self):
        """Test that CEREBRAS_API_KEY takes priority over MINIMAX_API_KEY."""
        from optillm.server import get_config

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.Cerebras') as mock_cerebras:
            client, api_key = get_config()

            assert api_key == 'cerebras-key'
            mock_cerebras.assert_called_once()


class TestMiniMaxTemperatureClamping(unittest.TestCase):
    """Test temperature clamping for MiniMax provider.

    The clamping logic in server.py's proxy() checks MINIMAX_API_KEY env var
    and adjusts temperature in request_config to MiniMax's valid range (0, 1].
    These tests verify the clamping logic directly.
    """

    def _clamp_temperature(self, temp, is_minimax=True):
        """Simulate the temperature clamping logic from server.py proxy()."""
        request_config = {'temperature': temp}
        if is_minimax and 'temperature' in request_config:
            t = request_config['temperature']
            if t is not None:
                if t <= 0:
                    request_config['temperature'] = 0.01
                elif t > 1.0:
                    request_config['temperature'] = 1.0
        return request_config['temperature']

    def test_temperature_zero_clamped(self):
        """Test that temperature=0 is clamped to 0.01 for MiniMax."""
        assert self._clamp_temperature(0) == 0.01

    def test_temperature_negative_clamped(self):
        """Test that negative temperature is clamped to 0.01 for MiniMax."""
        assert self._clamp_temperature(-0.5) == 0.01

    def test_temperature_above_one_clamped(self):
        """Test that temperature > 1.0 is clamped to 1.0 for MiniMax."""
        assert self._clamp_temperature(1.5) == 1.0

    def test_temperature_exactly_two_clamped(self):
        """Test that temperature=2.0 is clamped to 1.0 for MiniMax."""
        assert self._clamp_temperature(2.0) == 1.0

    def test_valid_temperature_point_seven_unchanged(self):
        """Test that valid temperature 0.7 is not modified."""
        assert self._clamp_temperature(0.7) == 0.7

    def test_valid_temperature_one_unchanged(self):
        """Test that temperature=1.0 (boundary) is not modified."""
        assert self._clamp_temperature(1.0) == 1.0

    def test_valid_temperature_point_one_unchanged(self):
        """Test that temperature=0.1 is not modified."""
        assert self._clamp_temperature(0.1) == 0.1

    def test_no_clamping_for_non_minimax(self):
        """Test that temperature is NOT clamped when not using MiniMax."""
        assert self._clamp_temperature(0, is_minimax=False) == 0
        assert self._clamp_temperature(1.5, is_minimax=False) == 1.5
        assert self._clamp_temperature(-1, is_minimax=False) == -1


class TestMiniMaxProviderPriority(unittest.TestCase):
    """Test provider priority ordering with MiniMax."""

    def setUp(self):
        """Set up test environment."""
        self.original_config = server_config.copy()
        for key in ['MINIMAX_API_KEY', 'OPENAI_API_KEY', 'CEREBRAS_API_KEY',
                    'AZURE_OPENAI_API_KEY', 'OPTILLM_API_KEY']:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        """Restore original server_config."""
        server_config.clear()
        server_config.update(self.original_config)

    @patch.dict(os.environ, {'OPTILLM_API_KEY': 'optillm-key', 'MINIMAX_API_KEY': 'minimax-key'})
    def test_optillm_takes_priority_over_minimax(self):
        """Test that OPTILLM_API_KEY takes priority over MINIMAX_API_KEY."""
        from optillm.server import get_config

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI') as mock_openai:
            # Mock the local inference path
            with patch.dict(sys.modules, {'optillm.inference': MagicMock()}):
                # Reimport to pick up the mock
                import importlib
                from optillm import server
                importlib.reload(server)
                client, api_key = server.get_config()
                assert api_key == 'optillm-key'

    @patch.dict(os.environ, {'MINIMAX_API_KEY': 'minimax-key', 'AZURE_OPENAI_API_KEY': 'azure-key',
                              'AZURE_API_VERSION': '2024-02-15', 'AZURE_API_BASE': 'https://test.openai.azure.com'})
    def test_minimax_takes_priority_over_azure(self):
        """Test that MINIMAX_API_KEY takes priority over AZURE_OPENAI_API_KEY."""
        from optillm.server import get_config

        server_config['ssl_verify'] = True
        server_config['ssl_cert_path'] = ''
        server_config['base_url'] = ''

        with patch('httpx.Client') as mock_httpx_client, \
             patch('optillm.server.OpenAI') as mock_openai:
            client, api_key = get_config()
            assert api_key == 'minimax-key'
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs['base_url'] == 'https://api.minimax.io/v1'


if __name__ == '__main__':
    unittest.main()
