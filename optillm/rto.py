import re
import logging
import optillm
from optillm import conversation_logger

logger = logging.getLogger(__name__)

def extract_code_from_prompt(text):
    pattern = r'```(?:[\w-]+)?\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        logger.warning("Could not extract code from prompt. Returning original text.")
        return text

def round_trip_optimization(system_prompt: str, initial_query: str, client, model: str, request_config: dict = None, request_id: str = None) -> str:
    rto_completion_tokens = 0

    # Extract max_tokens from request_config with default
    max_tokens = 4096
    if request_config:
        max_tokens = request_config.get('max_tokens', max_tokens)

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}]

    # Generate initial code (C1)
    provider_request = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "n": 1,
        "temperature": 0.1
    }
    response_c1 = client.chat.completions.create(**provider_request)
    
    # Log provider call
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response_c1.model_dump() if hasattr(response_c1, 'model_dump') else response_c1
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    
    if (response_c1 is None or
            not response_c1.choices or
            response_c1.choices[0].message.content is None or
            response_c1.choices[0].finish_reason == "length"):
        raise Exception("RTO: provider returned an empty, None, or truncated response for the initial code (C1)")
    c1 = response_c1.choices[0].message.content
    rto_completion_tokens += response_c1.usage.completion_tokens

    # Generate description of the code (Q2)
    messages.append({"role": "assistant", "content": c1})
    messages.append({"role": "user", "content": "Summarize or describe the code you just created. The summary should be in form of an instruction such that, given the instruction you can create the code yourself."})
    provider_request = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "n": 1,
        "temperature": 0.1
    }
    response_q2 = client.chat.completions.create(**provider_request)
    
    # Log provider call
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response_q2.model_dump() if hasattr(response_q2, 'model_dump') else response_q2
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    
    if (response_q2 is None or
            not response_q2.choices or
            response_q2.choices[0].message.content is None or
            response_q2.choices[0].finish_reason == "length"):
        raise Exception("RTO: provider returned an empty, None, or truncated response for the description (Q2)")
    q2 = response_q2.choices[0].message.content
    rto_completion_tokens += response_q2.usage.completion_tokens

    # Generate second code based on the description (C2)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": q2}]
    provider_request = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "n": 1,
        "temperature": 0.1
    }
    response_c2 = client.chat.completions.create(**provider_request)
    
    # Log provider call
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response_c2.model_dump() if hasattr(response_c2, 'model_dump') else response_c2
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    
    if (response_c2 is None or
            not response_c2.choices or
            response_c2.choices[0].message.content is None or
            response_c2.choices[0].finish_reason == "length"):
        raise Exception("RTO: provider returned an empty, None, or truncated response for the second code (C2)")
    c2 = response_c2.choices[0].message.content
    rto_completion_tokens += response_c2.usage.completion_tokens

    c1 = extract_code_from_prompt(c1)
    c2 = extract_code_from_prompt(c2)

    if c1.strip() == c2.strip():
        return c1, rto_completion_tokens

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Initial query: {initial_query}\n\nFirst generated code (C1):\n{c1}\n\nSecond generated code (C2):\n{c2}\n\nBased on the initial query and these two different code implementations, generate a final, optimized version of the code. Only respond with the final code, do not return anything else."}]
    provider_request = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "n": 1,
        "temperature": 0.1
    }
    response_c3 = client.chat.completions.create(**provider_request)
    
    # Log provider call
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response_c3.model_dump() if hasattr(response_c3, 'model_dump') else response_c3
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    
    if (response_c3 is None or
            not response_c3.choices or
            response_c3.choices[0].message.content is None or
            response_c3.choices[0].finish_reason == "length"):
        raise Exception("RTO: provider returned an empty, None, or truncated response for the final code (C3)")
    c3 = response_c3.choices[0].message.content
    rto_completion_tokens += response_c3.usage.completion_tokens

    return c3, rto_completion_tokens
