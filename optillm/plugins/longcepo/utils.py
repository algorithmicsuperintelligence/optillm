import logging
from typing import Callable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from .config import LongCepoConfig

logger = logging.getLogger(__name__)


class CBLog(dict):
    """Object for logging the number of LLM calls and tokens used in the pipeline"""

    __allowed_keys__ = {"total_tokens", "completion_tokens", "llm_calls"}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, key, value):
        if key not in self.__allowed_keys__:
            raise KeyError(
                f"Key '{key}' not allowed. Allowed keys: {self.__allowed_keys__}"
            )
        if not isinstance(value, int):
            raise TypeError(
                f"Value for '{key}' must be int, got {type(value).__name__}"
            )
        super().__setitem__(key, value)

    def update(self, other=None, **kwargs):
        updates = {}
        if other:
            if isinstance(other, dict):
                updates.update(other)
            else:
                updates.update(dict(other))
        updates.update(kwargs)

        for key, value in updates.items():
            if key not in self.__allowed_keys__:
                raise KeyError(
                    f"Key '{key}' not allowed. Allowed keys: {self.__allowed_keys__}"
                )
            if not isinstance(value, int):
                raise TypeError(
                    f"Value for '{key}' must be int, got {type(value).__name__}"
                )
            self[key] = self.get(key, 0) + value


def concurrent_map(
    gen_function: Callable,
    client,
    model: str,
    context_chunks: List[str],
    query: str,
    system_prompt: str,
    cb_log: CBLog,
    summaries_per_chunk: Optional[List[str]] = None,
    workers: int = 16,
) -> Tuple[List[str], CBLog]:
    """
    Runs `gen_function` concurrently over a list of context chunks.

    Args:
        gen_function (Callable): Function to call with each chunk and associated arguments.
        client: LLM API client.
        model (str): Base model name.
        context_chunks (List[str]): Input context chunks.
        query (str): User query.
        system_prompt (str): System prompt string.
        cb_log (CBLog): Log object for tracking model calls.
        summaries_per_chunk (Optional[List[str]]): Concatenated neighbor summaries for each chunk.
        workers (int): Number of threads to use.

    Returns:
        Tuple[List[str], CBLog]: List of responses (in original order) and updated log object.
    """
    result = [None] * len(context_chunks)
    def wrapped_gen_function(index, *args):
        return (index, gen_function(*args))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {}
        for idx, chunk in enumerate(context_chunks):
            args = [client, model, chunk, query, system_prompt]
            if summaries_per_chunk is not None:
                args.append(summaries_per_chunk[idx])
            future_to_idx[executor.submit(wrapped_gen_function, idx, *args)] = idx

        for future in as_completed(future_to_idx):
            try:
                index, (response, upd_log) = future.result()
                result[index] = response
                cb_log.update(upd_log)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
    return result, cb_log


def get_prompt_response(
    client,
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.7,
):
    """
    Helper function that sends a prompt to the chat-based LLM API and returns the generated response along with usage logging.

    Args:
        client: LLM API client.
        model (str): Base model name.
        prompt (str): The user prompt to send.
        system_prompt (str): System prompt string.
        max_tokens (int): Maximum number of tokens in the response.
        temperature (float): Sampling temperature for randomness (default: 0.7).
        top_p (float): Cumulative probability cutoff for token selection (default: 0.7).

    Returns:
        Tuple[str, CBLog]: The model's response text and a CBLog object tracking token usage.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        stream=False,
    )
    upd_log = CBLog(
        llm_calls=1,
        total_tokens=response.usage.total_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    return response.choices[0].message.content, upd_log


def loop_until_match(
    function: Callable, pattern_list: Tuple[str], num_attempts: int = 10
):
    """
    Repeatedly calls a function until its output matches one of the given patterns or max attempts is reached.

    Args:
        function (Callable): Function returning (answer: str, cb_log).
        pattern_list (Tuple[str]): Patterns to match in the answer.
        num_attempts (int): Max number of attempts (default: 10).

    Returns:
        Tuple[str, Any]: The matching answer and its corresponding log object.
    """
    correct_format = False
    for _ in range(num_attempts):
        answer, cb_log = function()

        for pattern in pattern_list:
            if pattern in answer:
                correct_format = True

        if correct_format:
            break

        logger.info("Wrong output formatting, retrying...")

    return answer, cb_log


def longcepo_init(
    initial_query: str,
) -> Tuple[str, str, PreTrainedTokenizerBase, CBLog, LongCepoConfig]:
    """
    Initializes context, query, tokenizer, logging, and config from an input string.

    Args:
        initial_query (str): Input string containing context and query separated by a delimiter string.

    Returns:
        Tuple[str, str, PreTrainedTokenizerBase, CBLog, LongCepoConfig]:
        Parsed context, query, tokenizer instance, log object, and LongCePO config.
    """
    cb_log = CBLog()
    config = LongCepoConfig()
    context, query = initial_query.split(config.context_query_delimiter)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    return context.strip(), query.strip(), tokenizer, cb_log, config
