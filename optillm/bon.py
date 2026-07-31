import logging
from optillm import conversation_logger

logger = logging.getLogger(__name__)

def best_of_n_sampling(system_prompt: str, initial_query: str, client, model: str, n: int = 3, request_config: dict = None, request_id: str = None) -> str:
    bon_completion_tokens = 0

    # Extract max_tokens from request_config with default
    max_tokens = 4096
    if request_config:
        max_tokens = request_config.get('max_tokens', max_tokens)

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}]
    
    completions = []
    
    try:
        # Try to generate n completions in a single API call using n parameter
        provider_request = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "n": n,
            "temperature": 1
        }
        response = client.chat.completions.create(**provider_request)

        # Log provider call
        if request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        # Check for valid response with None-checking
        if response is None or not response.choices:
            raise Exception("Response is None or has no choices")

        completions = [choice.message.content for choice in response.choices if choice.message.content is not None]
        logger.info(f"Generated {len(completions)} initial completions using n parameter. Tokens used: {response.usage.completion_tokens}")
        bon_completion_tokens += response.usage.completion_tokens

        # Check if any valid completions were generated
        if not completions:
            raise Exception("No valid completions generated (all were None)")

    except Exception as e:
        logger.warning(f"n parameter not supported by provider: {str(e)}")
        logger.info(f"Falling back to generating {n} completions one by one")
        
        # Fallback: Generate completions one by one in a loop
        for i in range(n):
            try:
                provider_request = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 1
                }
                response = client.chat.completions.create(**provider_request)

                # Log provider call
                if request_id:
                    response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                    conversation_logger.log_provider_call(request_id, provider_request, response_dict)

                # Check for valid response with None-checking
                if (response is None or
                    not response.choices or
                    response.choices[0].message.content is None or
                    response.choices[0].finish_reason == "length"):
                    logger.warning(f"Completion {i+1}/{n} truncated or empty, skipping")
                    continue

                completions.append(response.choices[0].message.content)
                bon_completion_tokens += response.usage.completion_tokens
                logger.debug(f"Generated completion {i+1}/{n}")
                
            except Exception as fallback_error:
                logger.error(f"Error generating completion {i+1}: {str(fallback_error)}")
                continue
        
        if not completions:
            logger.error("Failed to generate any completions")
            return "Error: Could not generate any completions", 0
        
        logger.info(f"Generated {len(completions)} completions using fallback method. Total tokens used: {bon_completion_tokens}")

    # Double-check we have completions before rating
    if not completions:
        logger.error("No completions available for rating")
        return "Error: Could not generate any completions", bon_completion_tokens

    # Rate the completions
    rating_messages = messages.copy()
    rating_messages.append({"role": "system", "content": "Rate the following responses on a scale from 0 to 10, where 0 is poor and 10 is excellent. Consider factors such as relevance, coherence, and helpfulness. Respond with only a number."})

    ratings = []
    for completion in completions:
        rating_messages.append({"role": "assistant", "content": completion})
        rating_messages.append({"role": "user", "content": "Rate the above response:"})
        
        provider_request = {
            "model": model,
            "messages": rating_messages,
            "max_tokens": 256,
            "n": 1,
            "temperature": 0.1
        }
        rating_response = client.chat.completions.create(**provider_request)

        # Log provider call
        if request_id:
            response_dict = rating_response.model_dump() if hasattr(rating_response, 'model_dump') else rating_response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        bon_completion_tokens += rating_response.usage.completion_tokens

        # Check for valid response with None-checking
        if (rating_response is None or
            not rating_response.choices or
            rating_response.choices[0].message.content is None or
            rating_response.choices[0].finish_reason == "length"):
            logger.warning("Rating response truncated or empty, using default rating of 0")
            ratings.append(0)
        else:
            try:
                rating = float(rating_response.choices[0].message.content.strip())
                ratings.append(rating)
            except ValueError:
                ratings.append(0)
        
        rating_messages = rating_messages[:-2]
    
    best_index = ratings.index(max(ratings))
    return completions[best_index], bon_completion_tokens
