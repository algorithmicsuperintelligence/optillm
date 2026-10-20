"""
Unit tests for SSRF protection in the readurls plugin.

Tests verify that the readurls plugin refuses to fetch non-public
destinations (private, loopback, link-local, reserved, multicast,
unspecified addresses), re-validates every redirect hop, fails closed on
DNS errors, and that operators can explicitly opt out via
server_config['readurls_allow_internal'].

All tests are fully offline: DNS resolution is either exercised only on
literal IPs (which libc resolves without network access) or mocked.
"""

import unittest
from unittest.mock import MagicMock, patch
import socket
import sys
import os

# Add parent directory to path to import optillm modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm import server_config


METADATA_URL = 'http://169.254.169.254/latest/meta-data/iam/security-credentials/'


def _addrinfo(ip):
    """Build a getaddrinfo-shaped result for a single IP string."""
    af = socket.AF_INET6 if ':' in ip else socket.AF_INET
    return [(af, socket.SOCK_STREAM, 6, '', (ip, 0))]


def _getaddrinfo_by_host(host_to_ip):
    """Return a getaddrinfo stand-in: known hosts resolve via the mapping,
    anything else (e.g. literal IPs) resolves to itself."""
    def fake_getaddrinfo(host, *args, **kwargs):
        return _addrinfo(host_to_ip.get(host, host))
    return fake_getaddrinfo


class TestIsBlockedHost(unittest.TestCase):
    """Unit tests for the _is_blocked_host guard."""

    def test_blocks_loopback_ipv4(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://127.0.0.1/index.html'))

    def test_blocks_metadata_ip(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host(METADATA_URL))

    def test_blocks_private_10(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://10.0.0.1/'))

    def test_blocks_private_192_168(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://192.168.1.1/'))

    def test_blocks_private_172_16(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://172.16.0.5/'))

    def test_blocks_unspecified(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://0.0.0.0/'))

    def test_blocks_ipv6_loopback(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://[::1]/'))

    def test_blocks_ipv6_link_local(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://[fe80::1]/'))

    def test_blocks_localhost(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        with patch('optillm.plugins.readurls_plugin.socket.getaddrinfo',
                   return_value=_addrinfo('127.0.0.1')):
            self.assertTrue(_is_blocked_host('http://localhost/secret'))

    def test_blocks_missing_hostname(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http:///path'))

    def test_blocks_unresolvable_host(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        with patch('optillm.plugins.readurls_plugin.socket.getaddrinfo',
                   side_effect=socket.gaierror('name resolution failure')):
            self.assertTrue(_is_blocked_host('http://internal.example.invalid/'))

    def test_allows_public_literal_ip(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertFalse(_is_blocked_host('http://93.184.216.34/'))

    def test_allows_public_resolved_host(self):
        from optillm.plugins.readurls_plugin import _is_blocked_host
        with patch('optillm.plugins.readurls_plugin.socket.getaddrinfo',
                   return_value=_addrinfo('93.184.216.34')):
            self.assertFalse(_is_blocked_host('https://example.com/article'))

    def test_blocks_cgnat_shared_range(self):
        """100.64.0.0/10 (carrier-grade NAT, e.g. Tailscale) is not globally
        routable even though it is outside the classic private ranges."""
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://100.64.0.1/'))
        self.assertTrue(_is_blocked_host('http://100.100.100.100/'))

    def test_blocks_multicast(self):
        """Multicast ranges are marked global by ipaddress but must be blocked."""
        from optillm.plugins.readurls_plugin import _is_blocked_host
        self.assertTrue(_is_blocked_host('http://224.0.0.1/'))
        self.assertTrue(_is_blocked_host('http://[ff02::1]/'))

    def test_blocks_empty_resolution(self):
        """An empty resolution result must fail closed, not be allowed."""
        from optillm.plugins.readurls_plugin import _is_blocked_host
        with patch('optillm.plugins.readurls_plugin.socket.getaddrinfo',
                   return_value=[]):
            self.assertTrue(_is_blocked_host('http://example.com/'))


class TestFetchWebpageContentGuard(unittest.TestCase):
    """Behavior tests: fetch_webpage_content refuses non-public URLs."""

    def setUp(self):
        self.original_config = server_config.copy()

    def tearDown(self):
        server_config.clear()
        server_config.update(self.original_config)

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_fetch_blocks_metadata_url(self, mock_requests_get):
        """The fetcher must short-circuit before issuing any request."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content

        result = fetch_webpage_content(METADATA_URL)

        self.assertIn('Error fetching content', result)
        self.assertIn('blocked non-public URL', result)
        mock_requests_get.assert_not_called()

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_fetch_allows_public_url(self, mock_requests_get):
        """Public URLs are still fetched normally."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content

        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Public content</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        with patch('optillm.plugins.readurls_plugin.socket.getaddrinfo',
                   return_value=_addrinfo('93.184.216.34')):
            result = fetch_webpage_content('https://example.com/article')

        self.assertEqual(result, 'Public content')
        mock_requests_get.assert_called_once()

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_fetch_blocks_redirect_to_internal(self, mock_requests_get):
        """A public URL redirecting to an internal address must be blocked."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content

        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {'Location': METADATA_URL}
        redirect_response.raise_for_status = MagicMock()

        final_response = MagicMock()
        final_response.content = b'<html><body><p>Leaked</p></body></html>'
        final_response.raise_for_status = MagicMock()
        mock_requests_get.side_effect = [redirect_response, final_response]

        with patch('optillm.plugins.readurls_plugin.socket.getaddrinfo',
                   side_effect=_getaddrinfo_by_host({'example.com': '93.184.216.34'})):
            result = fetch_webpage_content('https://example.com/redirect')

        self.assertIn('blocked non-public URL', result)
        self.assertNotIn('Leaked', result)
        # Only the initial public request may have been issued
        self.assertEqual(mock_requests_get.call_count, 1)

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_fetch_follows_public_redirect(self, mock_requests_get):
        """Redirects between public hosts keep working."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content

        redirect_response = MagicMock()
        redirect_response.status_code = 301
        redirect_response.headers = {'Location': 'https://example.com/moved'}
        final_response = MagicMock()
        final_response.content = b'<html><body><p>Moved content</p></body></html>'
        mock_requests_get.side_effect = [redirect_response, final_response]

        with patch('optillm.plugins.readurls_plugin.socket.getaddrinfo',
                   return_value=_addrinfo('93.184.216.34')):
            result = fetch_webpage_content('https://example.com/old')

        self.assertEqual(result, 'Moved content')
        self.assertEqual(mock_requests_get.call_count, 2)
        self.assertEqual(mock_requests_get.call_args_list[1][0][0],
                         'https://example.com/moved')

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_fetch_allow_internal_opt_out(self, mock_requests_get):
        """Operators can explicitly allow internal fetches."""
        from optillm.plugins.readurls_plugin import fetch_webpage_content

        server_config['readurls_allow_internal'] = True

        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Internal docs</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        result = fetch_webpage_content('http://127.0.0.1:8080/internal-docs')

        self.assertEqual(result, 'Internal docs')
        mock_requests_get.assert_called_once()


class TestReadurlsRun(unittest.TestCase):
    """run()-level tests with mixed public and internal URLs."""

    def setUp(self):
        self.original_config = server_config.copy()

    def tearDown(self):
        server_config.clear()
        server_config.update(self.original_config)

    @patch('optillm.plugins.readurls_plugin.requests.get')
    def test_run_blocks_internal_and_keeps_public(self, mock_requests_get):
        from optillm.plugins.readurls_plugin import run

        mock_response = MagicMock()
        mock_response.content = b'<html><body><p>Public page text</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        query = f"Summarize {METADATA_URL} and https://example.com/page"
        with patch('optillm.plugins.readurls_plugin.socket.getaddrinfo',
                   side_effect=_getaddrinfo_by_host({'example.com': '93.184.216.34'})):
            modified_query, _ = run("", query)

        # Only the public URL reached the network layer
        self.assertEqual(mock_requests_get.call_count, 1)
        self.assertIn('blocked non-public URL', modified_query)
        self.assertIn('Public page text', modified_query)


if __name__ == '__main__':
    unittest.main()
