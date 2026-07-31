from functools import partial
from typing import Tuple, List

from .utils import (
    CBLog,
    LongCepoConfig,
    get_prompt_response,
    concurrent_map,
    logger,
    loop_until_match,
)
from .chunking import (
    chunk_context,
    get_prompt_length,
)

def format_chunk_list(chunk_list):
    return [
    f"Information of Chunk {index}:\n{doc}\n" for index, doc in enumerate(chunk_list)
]


def remove_chunks(chunks: List[str], irrelevance_tags: Tuple[str]) -> List[str]:
    """
    Filter out chunks that contain at least one of irrelevance tags.
    """
    new_chunks = []
    for chunk in chunks:
        flag = False
        for tag in irrelevance_tags:
            if tag.upper() in chunk.upper():
                flag = True
                break
        if not flag:
            new_chunks.append(chunk)
    return new_chunks


def mapreduce(
    system_prompt: str,
    query: str,
    context: str,
    qa_history: str,
    client,
    model: str,
    tokenizer,
    longcepo_config: LongCepoConfig,
    cb_log: CBLog,
    answer_tags: Tuple[str] = ("Answer:", "**Answer**:", "**Answer**"),
    irrelevance_tags: Tuple[str] = ("[NO INFORMATION]",),
) -> Tuple[str, CBLog]:
    """
    Executes a MapReduce-style inference pipeline to answer a query from long context.

    The function splits the input context into chunks, summarizes and evaluates each with the model (Map),
    collapses intermediate answers to reduce redundancy, and then generates a final answer (Reduce).
    Irrelevant responses are filtered based on `irrelevance_tags`.

    Args:
        system_prompt (str): System prompt string.
        query (str): User query.
        context (str): Long-form input context to process.
        qa_history (str): QA history string for prompt injection.
        client: LLM API client.
        model (str): Base model name.
        tokenizer: Tokenizer to compute token lengths.
        longcepo_config (LongCepoConfig): Config with hyper-parameters and token limits.
        cb_log (CBLog): Log object for tracking model calls.
        answer_tags (Tuple[str]): Tags used to extract the final answer from model output.
        irrelevance_tags (Tuple[str]): Tags used to identify and remove irrelevant outputs.

    Returns:
        Tuple[str, CBLog]: Final extracted answer and updated log object.
    """

    logger.info(f"MapReduce query: {query}")

    qa_history_stub = (
        f"\n\nAnswers to related questions:\n\n{qa_history}" if qa_history else ""
    )

    context_chunks = chunk_context(context, longcepo_config.chunk_size, tokenizer)

    # Get short summaries of each chunk
    def fetch_chunk_summary(client, model, chunk, query, system_prompt):
        return get_prompt_response(
            client,
            model,
            longcepo_config.summary_prompt.format(question=query, context=chunk),
            system_prompt,
            max_tokens=longcepo_config.max_output_tokens_summary,
            temperature=longcepo_config.temperature_map,
        )

    summaries, cb_log = concurrent_map(
        fetch_chunk_summary,
        client,
        model,
        context_chunks,
        query,
        system_prompt,
        cb_log,
    )
    num_summaries = longcepo_config.num_neighbor_summaries
    # For each chunk position, get a neighborhood of `num_summaries` before and after the position
    summaries_per_chunk = [
        "\n\n".join(
            summaries[
                max(0, (summary_idx - num_summaries)) : min(
                    len(summaries) - 1, (summary_idx + num_summaries)
                )
            ]
        )
        for summary_idx in range(len(summaries))
    ]

    # Map stage
    def fetch_map_response(client, model, chunk, query, system_prompt, summary):
        return get_prompt_response(
            client,
            model,
            longcepo_config.map_prompt.format(
                question=query,
                context=chunk,
                summary=summary,
                qa_history_stub=qa_history_stub,
            ),
            system_prompt,
            max_tokens=longcepo_config.max_output_tokens,
            temperature=longcepo_config.temperature_map,
        )

    result, cb_log = concurrent_map(
        fetch_map_response,
        client,
        model,
        context_chunks,
        query,
        system_prompt,
        cb_log,
        summaries_per_chunk=summaries_per_chunk,
    )
    result = remove_chunks(result, irrelevance_tags)
    if not result:
        return "No information", cb_log

    logger.info(
        f"Removed {len(context_chunks) - len(result)} chunks from total {len(context_chunks)} chunks"
    )

    # Collapse stage
    result, cb_log = collapse_chunks(
        client,
        model,
        result,
        query,
        system_prompt,
        qa_history_stub,
        tokenizer,
        cb_log,
        longcepo_config,
        irrelevance_tags,
    )
    if not result:
        return "No information", cb_log

    # Reduce stage
    prompt = longcepo_config.reduce_prompt.format(
        question=query,
        context=format_chunk_list(result),
        qa_history_stub=qa_history_stub,
    )
    gen_fn = partial(
        get_prompt_response,
        client=client,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=longcepo_config.max_output_tokens,
        temperature=longcepo_config.temperature_reduce,
    )
    reduce_result, upd_log = loop_until_match(gen_fn, answer_tags,)
    cb_log.update(upd_log)

    final_answer = reduce_result
    for answer_tag in answer_tags:
        if answer_tag in reduce_result:
            final_answer = reduce_result.split(answer_tag)[-1].strip()
            break

    return final_answer, cb_log


def collapse_chunks(
    client,
    model: str,
    context_chunks: List[str],
    query: str,
    system_prompt: str,
    qa_history_stub: str,
    tokenizer,
    cb_log: CBLog,
    longcepo_config: LongCepoConfig,
    irrelevance_tags: Tuple[str] = ("[NO INFORMATION]",),
) -> Tuple[List[str], CBLog]:
    """
    Collapses context chunk pairs in sliding window until the total token count fits within the context window.

    Args:
        client: LLM API client.
        model (str): Base model name.
        context_chunks (List[str]): Input context chunks.
        query (str): User query.
        system_prompt (str): System prompt string.
        qa_history_stub (str): QA history prefix.
        tokenizer: Tokenizer to compute token lengths.
        cb_log (CBLog): Log object for tracking model calls.
        longcepo_config (LongCepoConfig): Config with hyper-parameters and token limits.

    Returns:
        Tuple[List[str], CBLog]: Final context chunks and updated logs.
    """
    num_tokens = get_prompt_length(format_chunk_list(context_chunks), tokenizer)
    token_budget = (
        longcepo_config.max_context_window
        - get_prompt_length(longcepo_config.reduce_prompt, tokenizer)
        - longcepo_config.max_output_tokens
    )
    logger.info(f"Pre-collapse length of chunks {num_tokens}, allowed {token_budget}")

    def fetch_collapse_response(client, model, docs, query, system_prompt):
        if len(docs) == 1:
            return docs[0], CBLog()
        return get_prompt_response(
            client,
            model,
            longcepo_config.collapse_prompt.format(
                question=query,
                context="\n\n".join(docs),
                qa_history_stub=qa_history_stub,
            ),
            system_prompt,
            max_tokens=longcepo_config.max_output_tokens,
            temperature=longcepo_config.temperature_collapse,
        )

    merge_pair_idx = 0
    collapse_step = 0
    while num_tokens is not None and num_tokens > token_budget:
        logger.info(f"Length at collapse stage {collapse_step}: {collapse_step}")

        if len(context_chunks) == 1:
            logger.info(f"Post-collapse length of chunks {num_tokens}")
            return context_chunks, cb_log

        # Merge chunk pair in a sliding window (merge_pair_idx:merge_pair_idx+2)
        chunk_groups = (
            [(context_chunks[i],) for i in range(merge_pair_idx)]
            + [(context_chunks[merge_pair_idx], context_chunks[merge_pair_idx + 1])]
            + [
                (context_chunks[i],)
                for i in range(merge_pair_idx + 2, len(context_chunks))
            ]
        )
        context_chunks, cb_log = concurrent_map(
            fetch_collapse_response,
            client,
            model,
            chunk_groups,
            query,
            system_prompt,
            cb_log,
        )
        context_chunks = remove_chunks(context_chunks, irrelevance_tags)
        merge_pair_idx = (merge_pair_idx + 1) % max(len(context_chunks) - 1, 1)
        num_tokens = get_prompt_length(format_chunk_list(context_chunks), tokenizer)
        collapse_step += 1

    logger.info(f"Post-collapse length of chunks {num_tokens}")
    return context_chunks, cb_log
