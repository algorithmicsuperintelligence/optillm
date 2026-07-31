"""
Deep Research Plugin - OptILLM Plugin Interface

This plugin implements the Test-Time Diffusion Deep Researcher (TTD-DR) algorithm
from the paper "Deep Researcher with Test-Time Diffusion".

Paper: https://arxiv.org/abs/2507.16075v1

The plugin orchestrates web search, URL fetching, and memory synthesis to provide
comprehensive research responses using an iterative refinement approach.
"""

from typing import Tuple, Dict, Optional
from optillm.plugins.deep_research import DeepResearcher

SLUG = "deep_research"


class DeepResearchClientWrapper:
    """Wrapper that adds extended timeout support for different client types"""
    
    def __init__(self, client, timeout=1800.0, max_retries=0):
        self.client = client
        self.timeout = timeout
        self.max_retries = max_retries
        self.client_type = self._detect_client_type()
        self.chat = self.Chat(self)
    
    def _detect_client_type(self):
        """Detect the type of client based on class name"""
        class_name = self.client.__class__.__name__
        
        # Check for OpenAI-compatible clients (OpenAI, Cerebras, AzureOpenAI)
        if 'OpenAI' in class_name or 'Cerebras' in class_name:
            return 'openai_compatible'
        # Check for LiteLLM wrapper
        elif 'LiteLLMWrapper' in class_name:
            return 'litellm'
        # All other clients (OptILLM inference, etc.)
        else:
            return 'other'
    
    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)
        
        class Completions:
            def __init__(self, parent):
                self.parent = parent
            
            def create(self, **kwargs):
                """Create completion with appropriate timeout handling"""
                if self.parent.client_type == 'openai_compatible':
                    # For OpenAI-compatible clients, recreate with timeout
                    try:
                        # Import here to avoid circular dependencies
                        if 'Cerebras' in self.parent.client.__class__.__name__:
                            from cerebras.cloud.sdk import Cerebras
                            custom_client = Cerebras(
                                api_key=self.parent.client.api_key,
                                base_url=getattr(self.parent.client, 'base_url', None),
                                timeout=self.parent.timeout,
                                max_retries=self.parent.max_retries
                            )
                        else:
                            # OpenAI or AzureOpenAI
                            # Get existing http_client to preserve SSL settings
                            existing_http_client = getattr(self.parent.client, '_client', None)

                            if 'Azure' in self.parent.client.__class__.__name__:
                                from openai import AzureOpenAI
                                # AzureOpenAI has different parameters
                                custom_client = AzureOpenAI(
                                    api_key=self.parent.client.api_key,
                                    api_version=getattr(self.parent.client, 'api_version', None),
                                    azure_endpoint=getattr(self.parent.client, 'azure_endpoint', None),
                                    azure_ad_token_provider=getattr(self.parent.client, 'azure_ad_token_provider', None),
                                    timeout=self.parent.timeout,
                                    max_retries=self.parent.max_retries,
                                    http_client=existing_http_client
                                )
                            else:
                                from openai import OpenAI
                                custom_client = OpenAI(
                                    api_key=self.parent.client.api_key,
                                    base_url=getattr(self.parent.client, 'base_url', None),
                                    timeout=self.parent.timeout,
                                    max_retries=self.parent.max_retries,
                                    http_client=existing_http_client
                                )
                        return custom_client.chat.completions.create(**kwargs)
                    except Exception as e:
                        # If recreation fails, use original client
                        print(f"⚠️ Warning: Could not create custom client with timeout: {str(e)}")
                        return self.parent.client.chat.completions.create(**kwargs)
                
                elif self.parent.client_type == 'litellm':
                    # For LiteLLM, add timeout to the call
                    kwargs['timeout'] = self.parent.timeout
                    return self.parent.client.chat.completions.create(**kwargs)
                
                else:
                    # For other clients (like OptILLM), just pass through
                    # They handle timeouts internally
                    print(f"ℹ️ Using original client (type: {self.parent.client.__class__.__name__}) without timeout modification")
                    return self.parent.client.chat.completions.create(**kwargs)


def run(system_prompt: str, initial_query: str, client, model: str, request_config: Optional[Dict] = None) -> Tuple[str, int]:
    """
    Deep Research plugin implementing TTD-DR (Test-Time Diffusion Deep Researcher)
    
    This plugin orchestrates web search, URL fetching, and memory synthesis to provide
    comprehensive research responses using an iterative refinement approach.
    
    Based on: "Deep Researcher with Test-Time Diffusion" 
    https://arxiv.org/abs/2507.16075v1
    
    Args:
        system_prompt: System prompt for the conversation
        initial_query: User's research query
        client: OpenAI client for LLM calls
        model: Model name to use for synthesis
        request_config: Optional configuration dict with keys:
            - max_iterations: Maximum research iterations (default: 5)
            - max_sources: Maximum web sources per search (default: 30)
    
    Returns:
        Tuple of (comprehensive_research_response, total_completion_tokens)
    """
    # Parse configuration
    config = request_config or {}
    max_iterations = config.get("max_iterations", 5)  # Default to 5 iterations for faster results
    max_sources = config.get("max_sources", 30)  # Balanced for comprehensive coverage
    
    # Validate inputs
    if not initial_query.strip():
        return "Error: No research query provided", 0
    
    if not client:
        return "Error: No LLM client provided for research synthesis", 0
    
    # Create a wrapped client with extended timeout for deep research
    # Deep research can take a long time, so we need 30 minutes timeout and no retries
    wrapped_client = DeepResearchClientWrapper(client, timeout=1800.0, max_retries=0)
    
    # Initialize researcher with wrapped client
    researcher = DeepResearcher(
        client=wrapped_client,
        model=model,
        max_iterations=max_iterations,
        max_sources=max_sources
    )
    
    try:
        # Perform deep research
        result, total_tokens = researcher.research(system_prompt, initial_query)
        return result, total_tokens
        
    except Exception as e:
        error_message = f"Deep research failed: {str(e)}"
        return error_message, 0