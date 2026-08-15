import ipaddress
import re
import socket
from typing import Tuple, List, Optional
import requests
from requests.adapters import HTTPAdapter
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from optillm import __version__, server_config

SLUG = "readurls"


def resolve_public_ip(url: str) -> Optional[str]:
    """Resolve *url* once and return a pinned public IP address, if safe."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        return None

    try:
        addresses = socket.getaddrinfo(
            parsed_url.hostname, None, type=socket.SOCK_STREAM
        )
    except (socket.gaierror, ValueError):
        return None

    try:
        ip_addresses = [ipaddress.ip_address(address[4][0]) for address in addresses]
    except ValueError:
        return None

    if not ip_addresses or not all(address.is_global for address in ip_addresses):
        return None
    return str(ip_addresses[0])


class PinnedIPAdapter(HTTPAdapter):
    """Connect to a validated IP while retaining the original HTTPS hostname."""

    def __init__(self, verified_ip: str, hostname: str):
        self.verified_ip = verified_ip
        self.hostname = hostname
        super().__init__()

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        host_params, pool_kwargs = self.build_connection_pool_key_attributes(
            request, verify, cert
        )
        host_params["host"] = self.verified_ip
        if host_params["scheme"] == "https":
            pool_kwargs["assert_hostname"] = self.hostname
            pool_kwargs["server_hostname"] = self.hostname
        request.headers["Host"] = urlparse(request.url).netloc
        return self.poolmanager.connection_from_host(
            **host_params, pool_kwargs=pool_kwargs
        )


def is_safe_url(url: str) -> bool:
    return resolve_public_ip(url) is not None


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

        current_url = url
        for _ in range(5):
            verified_ip = resolve_public_ip(current_url)
            parsed_url = urlparse(current_url)
            if not verified_ip or not parsed_url.hostname:
                return "Error fetching content: blocked unsafe URL"

            session = requests.Session()
            session.trust_env = False
            session.mount(
                f"{parsed_url.scheme}://",
                PinnedIPAdapter(verified_ip, parsed_url.hostname),
            )
            response = session.get(
                current_url,
                headers=headers,
                timeout=10,
                verify=verify,
                allow_redirects=False,
            )
            if not response.is_redirect:
                break

            location = response.headers.get("Location")
            if not location:
                return "Error fetching content: redirect missing Location header"
            current_url = urljoin(current_url, location)
        else:
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
