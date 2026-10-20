import re
from typing import Tuple, List, Optional
import requests
import os
import ipaddress
import socket
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from optillm import __version__, server_config

SLUG = "readurls"

# Redirect statuses followed manually so each hop can be re-checked by the
# SSRF guard before the next request is issued
REDIRECT_STATUS_CODES = frozenset((301, 302, 303, 307, 308))
MAX_REDIRECTS = 5


def _is_blocked_host(url: str) -> bool:
    """
    SSRF guard: return True when the URL does not point at a globally
    reachable public address (loopback, private, link-local, reserved,
    CGNAT/shared space, multicast or unspecified), or when the host cannot
    be resolved (fail-closed).

    Known limitation (DNS rebinding): the host is resolved here and the HTTP
    client performs a second resolution when it opens the connection, so a
    domain under attacker control could rebind to an internal address between
    the two lookups. For high-security deployments, additionally restrict the
    network egress of the optillm process instead of relying on this check
    alone.
    """
    try:
        hostname = urlparse(url).hostname
        if not hostname:
            return True

        # Resolve all addresses for the host and block the URL if any of
        # them is non-public
        addr_infos = socket.getaddrinfo(hostname, None)
        if not addr_infos:
            # No resolution results: fail closed
            return True
        for addr_info in addr_infos:
            ip = ipaddress.ip_address(addr_info[4][0])
            # ipaddress marks globally-routable multicast ranges as global,
            # so multicast must be blocked explicitly. CGNAT/shared space
            # (100.64.0.0/10, e.g. Tailscale) is not in the classic private
            # ranges but is not globally routable either.
            if ip.is_multicast or not ip.is_global:
                return True
        return False
    except (OSError, ValueError, UnicodeError):
        # Unresolvable or malformed hosts are blocked
        return True

def extract_urls(text: str) -> List[str]:
    # Updated regex pattern to be more precise
    url_pattern = re.compile(r'https?://[^\s\'"]+')
    
    # Find all matches
    urls = url_pattern.findall(text)
    
    # Clean up the URLs
    cleaned_urls = []
    for url in urls:
        # Remove trailing punctuation and quotes
        url = re.sub(r'[,\'\"\)\]]+$', '', url)
        cleaned_urls.append(url)
    
    return cleaned_urls

def fetch_webpage_content(url: str, max_length: int = 100000, verify_ssl: Optional[bool] = None, cert_path: Optional[str] = None) -> str:
    try:
        headers = {
            'User-Agent': f'optillm/{__version__} (https://github.com/codelion/optillm)'
        }

        # Use SSL configuration from server_config if not explicitly provided
        if verify_ssl is None:
            verify_ssl = server_config.get('ssl_verify', True)
        if cert_path is None:
            cert_path = server_config.get('ssl_cert_path', '')

        # Determine verify parameter for requests
        if not verify_ssl:
            verify = False
        elif cert_path:
            verify = cert_path
        else:
            verify = True

        # SSRF guard: resolve the host and refuse non-public destinations
        # before fetching. Redirects are followed manually so every hop is
        # re-checked, closing the bypass where a public URL redirects to an
        # internal address. Operators can explicitly allow internal fetches
        # via server_config['readurls_allow_internal'].
        allow_internal = server_config.get('readurls_allow_internal', False)
        current_url = url
        response = None
        for _ in range(MAX_REDIRECTS + 1):
            if not allow_internal and _is_blocked_host(current_url):
                return "Error fetching content: blocked non-public URL"
            response = requests.get(current_url, headers=headers, timeout=10, verify=verify, allow_redirects=False)
            if response.status_code not in REDIRECT_STATUS_CODES:
                break
            location = response.headers.get('Location')
            if not location:
                break
            current_url = urljoin(current_url, location)

        if response.status_code in REDIRECT_STATUS_CODES:
            return "Error fetching content: too many redirects"
        response.raise_for_status()
        
        # Make a soup 
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text from various elements
        text_elements = []
        
        # Prioritize content from main content tags
        for tag in ['article', 'main', 'div[role="main"]', '.main-content']:
            content = soup.select_one(tag)
            if content:
                text_elements.extend(content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table']))
                break
        
        # If no main content found, fall back to all headers, paragraphs, and tables
        if not text_elements:
            text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'table'])
        
        # Process all elements including tables
        content_parts = []
        
        for element in text_elements:
            if element.name == 'table':
                # Process table
                table_content = []
                
                # Get headers
                headers = element.find_all('th')
                if headers:
                    header_text = ' | '.join(header.get_text(strip=True) for header in headers)
                    table_content.append(header_text)
                
                # Get rows
                for row in element.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_text = ' | '.join(cell.get_text(strip=True) for cell in cells)
                        table_content.append(row_text)
                
                # Add table content with proper spacing
                content_parts.append('\n' + '\n'.join(table_content) + '\n')
            else:
                # Process regular text elements
                content_parts.append(element.get_text(strip=False))
        
        # Join all content
        text = ' '.join(content_parts)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove footnote superscripts in brackets
        text = re.sub(r"\[.*?\]+", '', text)
        
        # Truncate to max_length
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return text
    except Exception as e:
        return f"Error fetching content: {str(e)}"

def run(system_prompt, initial_query: str, client=None, model=None) -> Tuple[str, int]:
    urls = extract_urls(initial_query)
    # print(urls)
    modified_query = initial_query

    for url in urls:
        content = fetch_webpage_content(url)
        domain = urlparse(url).netloc
        modified_query = modified_query.replace(url, f"{url} [Content from {domain}: {content}]")
    # print(modified_query)
    return modified_query, 0
