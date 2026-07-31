import re
import time
import random
from typing import Tuple, List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import quote_plus

SLUG = "web_search"


class BrowserSessionManager:
    """
    Manages a single browser session across multiple searches.
    Implements context manager for automatic cleanup.
    """
    def __init__(self, headless: bool = False, timeout: int = 30):
        self.headless = headless
        self.timeout = timeout
        self._searcher = None
        self._search_count = 0
        self._session_start_time = None
    
    def __enter__(self):
        """Context manager entry - ensures browser is ready"""
        self.get_or_create_searcher()
        self._session_start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures browser cleanup"""
        self.close()
        return False  # Don't suppress exceptions
    
    def get_or_create_searcher(self) -> 'GoogleSearcher':
        """Get existing searcher or create a new one"""
        if self._searcher is None or self._searcher.driver is None:
            if self._searcher is None:
                print("🌐 Creating new browser session for research...")
            else:
                print("🔄 Recreating browser session (previous session invalidated)...")
            self._searcher = GoogleSearcher(
                headless=self.headless,
                timeout=self.timeout
            )
        return self._searcher
    
    def search(self, query: str, num_results: int = 10, delay_seconds: Optional[int] = None) -> List[Dict[str, str]]:
        """Perform a search using the managed browser session with automatic recovery"""
        try:
            searcher = self.get_or_create_searcher()
            self._search_count += 1
            session_duration = time.time() - self._session_start_time if self._session_start_time else 0
            print(f"🔍 Search #{self._search_count} in current session (instance: {id(self)}, duration: {session_duration:.1f}s): {query[:50]}...")
            return searcher.search(query, num_results, delay_seconds)
        except Exception as e:
            error_msg = str(e).lower()
            # Check for session-related errors
            if 'invalid session id' in error_msg or 'session deleted' in error_msg:
                print("⚠️  Browser session invalidated, attempting recovery...")
                # Invalidate current searcher
                if self._searcher:
                    try:
                        self._searcher.close()
                    except Exception:
                        pass  # Ignore errors during cleanup
                self._searcher = None
                
                # Try once more with a fresh session
                try:
                    searcher = self.get_or_create_searcher()
                    print("✅ New browser session created, retrying search...")
                    return searcher.search(query, num_results, delay_seconds)
                except Exception as retry_error:
                    print(f"❌ Session recovery failed: {str(retry_error)}")
                    return []  # Return empty results instead of crashing
            else:
                # For other errors, just log and return empty results
                print(f"❌ Search error: {str(e)}")
                return []
    
    def close(self):
        """Close the browser session"""
        if self._searcher is not None:
            try:
                self._searcher.close()
                if self._session_start_time:
                    duration = time.time() - self._session_start_time
                    print(f"🏁 Browser session closed after {self._search_count} searches ({duration:.1f}s)")
            except Exception as e:
                print(f"⚠️ Error closing browser session: {e}")
            finally:
                self._searcher = None
                self._search_count = 0
                self._session_start_time = None
    
    def is_active(self) -> bool:
        """Check if browser session is active"""
        return self._searcher is not None and self._searcher.driver is not None


class GoogleSearcher:
    def __init__(self, headless: bool = False, timeout: int = 30):
        self.timeout = timeout
        self.headless = headless
        self.driver = None
        self.setup_driver(headless)
    
    def setup_driver(self, headless: bool = False):
        """Setup Chrome driver with appropriate options"""
        try:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless")
            else:
                # Non-headless mode - position window for visibility
                chrome_options.add_argument("--window-size=1280,800")
                chrome_options.add_argument("--window-position=100,100")
            
            # Common options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # More human-like settings
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Use webdriver-manager to automatically manage ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
        except Exception as e:
            raise Exception(f"Failed to setup Chrome driver: {str(e)}")
    
    def detect_captcha(self) -> bool:
        """Detect if CAPTCHA is present on the page"""
        try:
            # Check for common CAPTCHA indicators
            page_source = self.driver.page_source.lower()
            captcha_indicators = [
                'recaptcha',
                'captcha',
                'are you a robot',
                'not a robot',
                'unusual traffic',
                'automated requests',
                'verify you\'re human',
                'verify that you\'re not a robot'
            ]
            
            for indicator in captcha_indicators:
                if indicator in page_source:
                    return True
            
            # Check for reCAPTCHA iframe
            try:
                self.driver.find_element(By.CSS_SELECTOR, "iframe[src*='recaptcha']")
                return True
            except Exception:
                pass
            
            # Check for CAPTCHA challenge div
            try:
                self.driver.find_element(By.ID, "captcha")
                return True
            except Exception:
                pass
            
            return False
        except Exception:
            return False
    
    def wait_for_captcha_resolution(self, max_wait: int = 300) -> bool:
        """Wait for CAPTCHA to be resolved with user confirmation"""
        print("🚨 CAPTCHA DETECTED! 🚨")
        print("Please solve the CAPTCHA in the browser window.")
        print("After solving the CAPTCHA, press ENTER here to continue...")
        
        if self.headless:
            print("ERROR: CAPTCHA detected in headless mode - cannot solve automatically")
            return False
        
        # Wait for user to press Enter after solving CAPTCHA
        try:
            input("Press ENTER after you have solved the CAPTCHA: ")
        except KeyboardInterrupt:
            print("\\nSearch cancelled by user")
            return False
        
        # Give a moment for the page to update after CAPTCHA resolution
        print("Checking if CAPTCHA has been resolved...")
        time.sleep(2)
        
        # Verify CAPTCHA is actually resolved
        for attempt in range(3):
            if not self.detect_captcha():
                print("✅ CAPTCHA resolved successfully!")
                return True
            else:
                print(f"CAPTCHA still detected (attempt {attempt + 1}/3)")
                if attempt < 2:
                    response = input("CAPTCHA still present. Try again? Press ENTER to continue or 'q' to quit: ")
                    if response.lower() == 'q':
                        return False
                    time.sleep(2)
        
        print("❌ CAPTCHA still not resolved after 3 attempts")
        return False
    
    def search(self, query: str, num_results: int = 10, delay_seconds: Optional[int] = None) -> List[Dict[str, str]]:
        """Perform Google search and return results"""
        if not self.driver:
            raise Exception("Chrome driver not initialized")
        
        try:
            print(f"Searching for: {query}")
            if not self.headless:
                print("Browser window opened")
            
            # First navigate to Google homepage
            self.driver.get("https://www.google.com")
            
            # Wait for page to load and check for CAPTCHA
            time.sleep(1)
            
            # Check if we hit a CAPTCHA immediately
            if self.detect_captcha():
                if not self.wait_for_captcha_resolution():
                    return []
            
            # Check for consent form or cookie banner and accept if present
            try:
                accept_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'I agree') or contains(text(), 'Agree')]")
                accept_button.click()
                time.sleep(1)
            except Exception:
                pass  # No consent form
            
            # Find search box and enter query
            try:
                # Try multiple selectors for the search box
                search_box = None
                for selector in [(By.NAME, "q"), (By.CSS_SELECTOR, "input[type='text']"), (By.CSS_SELECTOR, "textarea[name='q']")]:
                    try:
                        search_box = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located(selector)
                        )
                        break
                    except Exception:
                        continue
                
                if search_box:
                    # Use ActionChains for more reliable input
                    actions = ActionChains(self.driver)
                    actions.move_to_element(search_box)
                    actions.click()
                    actions.pause(0.5)
                    # Clear existing text
                    search_box.clear()
                    actions.send_keys(query)
                    actions.pause(0.5)
                    actions.send_keys(Keys.RETURN)
                    actions.perform()
                    
                    # Wait briefly for page to start loading
                    time.sleep(1)
                    
                    # Check for CAPTCHA after search submission
                    if self.detect_captcha():
                        if not self.wait_for_captcha_resolution():
                            return []
                else:
                    raise Exception("Could not find search box")
            except Exception:
                # Fallback to direct URL navigation
                print("Using direct URL navigation...")
                search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
                self.driver.get(search_url)
                time.sleep(1)
                
                # Check for CAPTCHA on direct navigation
                if self.detect_captcha():
                    if not self.wait_for_captcha_resolution():
                        return []
            
            # Wait for search results
            wait = WebDriverWait(self.driver, 10)
            try:
                wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.g, [data-sokoban-container], div[data-async-context]"))
                )
            except TimeoutException:
                # Check if it's a CAPTCHA page
                if self.detect_captcha():
                    if self.wait_for_captcha_resolution():
                        # Try waiting for results again after CAPTCHA
                        try:
                            wait.until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
                            )
                        except Exception:
                            print("No results found after CAPTCHA resolution")
                            return []
                    else:
                        return []
                else:
                    print("Timeout waiting for search results")
                    return []
            
            results = []
            
            # Apply delay AFTER search results are loaded (to prevent triggering anti-bot measures)
            if delay_seconds is None:
                delay_seconds = random.randint(4, 32)  # Updated range: 4-32 seconds
            
            if delay_seconds > 0:
                print(f"Applying {delay_seconds} second delay after search...")
                time.sleep(delay_seconds)
            
            print("Extracting search results...")
            
            # Wait for search results to be present
            try:
                print("Waiting for search results to load...")
                # Wait for either the search container or the results themselves
                WebDriverWait(self.driver, 10).until(
                    lambda driver: driver.find_elements(By.CSS_SELECTOR, "div.g") or 
                                   driver.find_element(By.ID, "search") or
                                   driver.find_elements(By.CSS_SELECTOR, "[data-sokoban-container]")
                )
            except TimeoutException:
                print("Timeout waiting for search results. Checking for CAPTCHA...")
                if self.detect_captcha():
                    if not self.wait_for_captcha_resolution():
                        return []
                    # Try waiting again after CAPTCHA
                    try:
                        WebDriverWait(self.driver, 10).until(
                            lambda driver: driver.find_elements(By.CSS_SELECTOR, "div.g")
                        )
                    except Exception:
                        print("Still no results after CAPTCHA resolution")
                        return []
                else:
                    print("No CAPTCHA detected, but timeout occurred - search may have failed")
                    return []
            
            # Debug: Print current URL and page title
            print(f"Current URL: {self.driver.current_url}")
            print(f"Page title: {self.driver.title}")
            
            # Extract search results - try multiple selectors
            search_results = []
            
            # First try the standard div.g selector
            search_results = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
            print(f"Found {len(search_results)} results with div.g")
            
            # If no results, try alternative selectors
            if not search_results:
                # Try finding any element with data-hveid attribute (Google result containers)
                all_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-hveid]")
                print(f"Found {len(all_elements)} elements with data-hveid")
                
                # Filter to only those that have both h3 and a tags
                for elem in all_elements:
                    try:
                        # Check if this element has both h3 and a link
                        h3 = elem.find_element(By.TAG_NAME, "h3")
                        link = elem.find_element(By.CSS_SELECTOR, "a[href]")
                        if h3 and link:
                            search_results.append(elem)
                    except Exception:
                        continue
                
                print(f"Filtered to {len(search_results)} valid result elements")
            
            if not search_results:
                print("No search results found with any method")
                # Debug: print some page source to see what we're getting
                print("Page source sample (first 500 chars):")
                print(self.driver.page_source[:500])
                return []
            
            # Limit processing to requested number of results
            results_to_process = min(len(search_results), num_results)
            print(f"Processing {results_to_process} results...")
            
            for i, result in enumerate(search_results[:results_to_process]):
                try:
                    # Skip if we already have enough results
                    if len(results) >= num_results:
                        break
                    
                    # Use the same extraction logic as getstars
                    try:
                        url = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        title = result.find_element(By.CSS_SELECTOR, "h3").text
                        
                        # Skip Google internal URLs
                        if not url or "google.com" in url:
                            continue
                            
                        # Try to get snippet
                        snippet = ""
                        try:
                            # Try multiple snippet selectors
                            snippet_selectors = [".VwiC3b", ".aCOpRe", ".IsZvec"]
                            for selector in snippet_selectors:
                                try:
                                    snippet_elem = result.find_element(By.CSS_SELECTOR, selector)
                                    if snippet_elem and snippet_elem.text:
                                        snippet = snippet_elem.text
                                        break
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        
                        # Add result
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet or "No description available"
                        })
                        
                        print(f"Extracted result {len(results)}: {title[:50]}...")
                        
                    except NoSuchElementException:
                        print(f"Failed to parse result {i+1}")
                        continue
                        
                except Exception:
                    # Skip problematic results
                    continue
            
            # Deduplicate results by URL
            seen_urls = set()
            unique_results = []
            for result in results:
                if result["url"] not in seen_urls:
                    seen_urls.add(result["url"])
                    unique_results.append(result)
            
            print(f"Successfully extracted {len(unique_results)} unique search results (from {len(results)} total)")
            
            return unique_results
            
        except TimeoutException as e:
            # Return empty results instead of raising
            print(f"Search timeout for query '{query}': {str(e)}")
            return []
        except WebDriverException as e:
            error_msg = str(e).lower()
            if 'invalid session id' in error_msg or 'session deleted' in error_msg:
                print(f"WebDriver session invalid: {str(e)}")
                # Invalidate the driver so next search creates a new one
                self.driver = None
            else:
                print(f"WebDriver error during search: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error during search: {str(e)}")
            return []
    
    def close(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def extract_search_queries(text: str) -> List[str]:
    """Extract potential search queries from the input text"""
    # Clean up common prefixes from chat messages
    text = text.strip()
    # Remove common role prefixes
    for prefix in ["User:", "user:", "User ", "user ", "Assistant:", "assistant:", "System:", "system:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Look for explicit search requests
    # Note: Removed period (.) from exclusion to allow queries like "Python 3.12" to work
    # Updated to require at least one non-whitespace character after the prefix
    search_patterns = [
        r"search for[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
        r"find information about[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
        r"look up[:\s]+(\S[^\n]*?)(?:\s*\n|$)", 
        r"research[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
    ]
    
    queries = []
    for pattern in search_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            cleaned = match.strip()
            # Remove trailing quotes (single or double)
            cleaned = cleaned.rstrip('"\'')
            # Remove leading quotes if they exist
            cleaned = cleaned.lstrip('"\'')
            # Only add non-empty queries
            if cleaned:
                queries.append(cleaned)
    
    # If no explicit patterns, use the text as a search query
    # Since the user explicitly invoked the web_search plugin, we should always search
    if not queries:
        # Check if this is a search command with empty query (e.g., "search for" with nothing after)
        search_prefixes = ["search for", "find information about", "look up", "research"]
        text_lower = text.lower().strip()

        # Don't use fallback if it's just a search prefix with nothing meaningful after
        is_empty_search = any(
            text_lower.startswith(prefix) and
            len(text_lower.replace(prefix, "").strip().strip('"\'')) < 2
            for prefix in search_prefixes
        )

        if not is_empty_search and text.strip():
            # User explicitly invoked web_search plugin - send the full query
            cleaned_query = text.strip()

            # Just basic cleanup: normalize whitespace (replace newlines with spaces)
            cleaned_query = ' '.join(cleaned_query.split())

            # Final validation - must have some meaningful content
            if cleaned_query and len(cleaned_query) >= 3:
                queries.append(cleaned_query)
    
    return queries

def format_search_results(query: str, results: List[Dict[str, str]]) -> str:
    """Format search results into readable text"""
    if not results:
        return f"No search results found for: {query}"
    
    formatted = f"Search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"{i}. **{result['title']}**\n"
        formatted += f"   URL: {result['url']}\n"
        if result['snippet']:
            formatted += f"   Summary: {result['snippet']}\n"
        formatted += "\n"
    
    return formatted

def run(system_prompt: str, initial_query: str, client=None, model: str = None, request_config: Optional[Dict] = None) -> Tuple[str, int]:
    """
    Web search plugin that uses Chrome to search Google and return results
    
    Args:
        system_prompt: System prompt for the conversation
        initial_query: User's query that may contain search requests
        client: OpenAI client (unused for this plugin)
        model: Model name (unused for this plugin) 
        request_config: Optional configuration dict with keys:
            - num_results: Number of search results (default: 10)
            - delay_seconds: Delay between searches in seconds (default: random 4-32)
                            Set to 0 to disable delays, or specify exact seconds
            - headless: Run browser in headless mode (default: False)
            - timeout: Browser timeout in seconds (default: 30)
            - session_manager: BrowserSessionManager instance for session reuse
    
    Returns:
        Tuple of (enhanced_query_with_search_results, completion_tokens)
    """
    # Parse configuration
    config = request_config or {}
    num_results = config.get("num_results", 10)
    delay_seconds = config.get("delay_seconds", None)  # None means random 4-32
    headless = config.get("headless", False)  # Default to non-headless
    timeout = config.get("timeout", 30)  # Standard timeout
    session_manager = config.get("session_manager", None)  # For session reuse
    
    # Extract search queries from the input
    search_queries = extract_search_queries(initial_query)
    
    if not search_queries:
        return initial_query, 0
    
    # Determine if we should manage the browser lifecycle
    own_session = session_manager is None
    
    try:
        # Use provided session manager or create temporary one
        if own_session:
            # Create temporary searcher for standalone use
            searcher = GoogleSearcher(headless=headless, timeout=timeout)
        
        enhanced_query = initial_query
        
        for query in search_queries:
            # Perform the search
            if session_manager:
                # Use session manager's search method
                results = session_manager.search(query, num_results=num_results, delay_seconds=delay_seconds)
            else:
                # Use temporary searcher
                results = searcher.search(query, num_results=num_results, delay_seconds=delay_seconds)
            
            # Format results
            if results:
                formatted_results = format_search_results(query, results)
                # Append results to the query
                enhanced_query = f"{enhanced_query}\n\n[Web Search Results]:\n{formatted_results}"
            else:
                # No results found - add a note
                enhanced_query = f"{enhanced_query}\n\n[Web Search Results]:\nNo results found for '{query}'. This may be due to network issues or search restrictions."
        
        return enhanced_query, 0
        
    except Exception as e:
        error_msg = f"Web search error: {str(e)}"
        enhanced_query = f"{initial_query}\n\n[Web Search Error]: {error_msg}"
        return enhanced_query, 0
        
    finally:
        # Only close if we created our own searcher
        if own_session and 'searcher' in locals():
            searcher.close()