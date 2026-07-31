import logging
from optillm import conversation_logger

logger = logging.getLogger(__name__)

def mixture_of_agents(system_prompt: str, initial_query: str, client, model: str, request_config: dict = None, request_id: str = None) -> str:
    logger.info(f"Starting mixture_of_agents function with model: {model}")
    moa_completion_tokens = 0

    # Extract max_tokens from request_config with default
    max_tokens = 4096
    if request_config:
        max_tokens = request_config.get('max_tokens', max_tokens)

    completions = []

    logger.debug(f"Generating initial completions for query: {initial_query}")
    
    try:
        # Try to generate 3 completions in a single API call using n parameter
        provider_request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}
            ],
            "max_tokens": max_tokens,
            "n": 3,
            "temperature": 1
        }
        
        response = client.chat.completions.create(**provider_request)

        # Convert response to dict for logging
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response

        # Log provider call if conversation logging is enabled
        if request_id:
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        # Check for valid response with None-checking
        if response is None or not response.choices:
            raise Exception("Response is None or has no choices")

        completions = [choice.message.content for choice in response.choices if choice.message.content is not None]
        moa_completion_tokens += response.usage.completion_tokens
        logger.info(f"Generated {len(completions)} initial completions using n parameter. Tokens used: {response.usage.completion_tokens}")

        # Check if any valid completions were generated
        if not completions:
            raise Exception("No valid completions generated (all were None)")

    except Exception as e:
        logger.warning(f"n parameter not supported by provider: {str(e)}")
        logger.info("Falling back to generating 3 completions one by one")
        
        # Fallback: Generate 3 completions one by one in a loop
        completions = []
        for i in range(3):
            try:
                provider_request = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_query}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 1
                }
                
                response = client.chat.completions.create(**provider_request)

                # Convert response to dict for logging
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response

                # Log provider call if conversation logging is enabled
                if request_id:
                    conversation_logger.log_provider_call(request_id, provider_request, response_dict)

                # Check for valid response with None-checking
                if (response is None or
                    not response.choices or
                    response.choices[0].message.content is None or
                    response.choices[0].finish_reason == "length"):
                    logger.warning(f"Completion {i+1}/3 truncated or empty, skipping")
                    continue

                completions.append(response.choices[0].message.content)
                moa_completion_tokens += response.usage.completion_tokens
                logger.debug(f"Generated completion {i+1}/3")
                
            except Exception as fallback_error:
                logger.error(f"Error generating completion {i+1}: {str(fallback_error)}")
                continue
        
        if not completions:
            logger.error("Failed to generate any completions")
            return "Error: Could not generate any completions", 0
        
        logger.info(f"Generated {len(completions)} completions using fallback method. Total tokens used: {moa_completion_tokens}")

    # Double-check we have at least one completion
    if not completions:
        logger.error("No completions available for processing")
        return "Error: Could not generate any completions", moa_completion_tokens

    # Handle case where fewer than 3 completions were generated
    if len(completions) < 3:
        original_count = len(completions)
        # Pad with the first completion to ensure we have 3
        while len(completions) < 3:
            completions.append(completions[0])
        logger.warning(f"Only generated {original_count} unique completions, padded to 3 for critique")
    
    logger.debug("Preparing critique prompt")
    critique_prompt = f"""
    Original query: {initial_query}

    I will present you with three candidate responses to the original query. Please analyze and critique each response, discussing their strengths and weaknesses. Provide your analysis for each candidate separately.

    Candidate 1:
    {completions[0]}

    Candidate 2:
    {completions[1]}

    Candidate 3:
    {completions[2]}

    Please provide your critique for each candidate:
    """

    logger.debug("Generating critiques")
    
    provider_request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critique_prompt}
        ],
        "max_tokens": 512,
        "n": 1,
        "temperature": 0.1
    }
    
    critique_response = client.chat.completions.create(**provider_request)

    # Convert response to dict for logging
    response_dict = critique_response.model_dump() if hasattr(critique_response, 'model_dump') else critique_response

    # Log provider call if conversation logging is enabled
    if request_id:
        conversation_logger.log_provider_call(request_id, provider_request, response_dict)

    # Check for valid response with None-checking
    if (critique_response is None or
        not critique_response.choices or
        critique_response.choices[0].message.content is None or
        critique_response.choices[0].finish_reason == "length"):
        logger.warning("Critique response truncated or empty, using generic critique")
        critiques = "All candidates show reasonable approaches to the problem."
    else:
        critiques = critique_response.choices[0].message.content

    moa_completion_tokens += critique_response.usage.completion_tokens
    logger.info(f"Generated critiques. Tokens used: {critique_response.usage.completion_tokens}")
    
    logger.debug("Preparing final prompt")
    final_prompt = f"""
    Original query: {initial_query}

    Based on the following candidate responses and their critiques, generate a final response to the original query.

    Candidate 1:
    {completions[0]}

    Candidate 2:
    {completions[1]}

    Candidate 3:
    {completions[2]}

    Critiques of all candidates:
    {critiques}

    Please provide a final, optimized response to the original query:
    """

    logger.debug("Generating final response")

    provider_request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ],
        "max_tokens": max_tokens,
        "n": 1,
        "temperature": 0.1
    }
    
    final_response = client.chat.completions.create(**provider_request)

    # Convert response to dict for logging
    response_dict = final_response.model_dump() if hasattr(final_response, 'model_dump') else final_response

    # Log provider call if conversation logging is enabled
    if request_id:
        conversation_logger.log_provider_call(request_id, provider_request, response_dict)

    moa_completion_tokens += final_response.usage.completion_tokens
    logger.info(f"Generated final response. Tokens used: {final_response.usage.completion_tokens}")

    # Check for valid response with None-checking
    if (final_response is None or
        not final_response.choices or
        final_response.choices[0].message.content is None or
        final_response.choices[0].finish_reason == "length"):
        logger.error("Final response truncated or empty. Consider increasing max_tokens.")
        # Return best completion if final response failed
        result = completions[0] if completions else "Error: Response was truncated due to token limit. Please increase max_tokens or max_completion_tokens."
    else:
        result = final_response.choices[0].message.content

    logger.info(f"Total completion tokens used: {moa_completion_tokens}")
    return result, moa_completion_tokens