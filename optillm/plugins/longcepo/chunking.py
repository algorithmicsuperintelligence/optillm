# Code modified from https://github.com/thunlp/LLMxMapReduce under Apache 2.0

import re
from typing import List

from .utils import logger


def get_prompt_length(prompt: str, tokenizer, no_special_tokens=False, **kwargs) -> int:
    """
    Returns the token length of a prompt using the given tokenizer.
    """
    if isinstance(prompt, list):
        prompt = "\n\n".join(prompt)
    if no_special_tokens:
        kwargs["add_special_tokens"] = False
    return len(tokenizer.encode(prompt, **kwargs))


def chunk_context(doc: str, chunk_size: int, tokenizer, separator="\n",) -> List[str]:
    """
    Splits a long document into token-limited chunks based on a separator, ensuring each chunk fits within `chunk_size`.

    Uses a greedy approach to accumulate text segments (split by `separator`) into chunks that fit within the
    token limit. If a segment alone exceeds the limit, it is recursively broken down using sentence-level
    splitting. Attempts to preserve natural boundaries while minimizing excessive chunking.

    Args:
        doc (str): Input document to split.
        chunk_size (int): Maximum number of tokens allowed per chunk.
        tokenizer: Tokenizer instance with `.encode()` method to compute token length.
        separator (str): Delimiter to split initial segments (default: newline).

    Returns:
        List[str]: List of non-empty, token-constrained document chunks.
    """
    paragraphs = doc.split(separator)
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    separator_len = get_prompt_length(separator, tokenizer, no_special_tokens=True)

    docs = []
    current_doc = []
    total = 0
    for paragraph in paragraphs:
        plen = get_prompt_length(paragraph, tokenizer, no_special_tokens=True)
        if total + plen + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
            if total > chunk_size:
                logger.info(
                    f"Created a chunk of size {total}, "
                    f"which is longer than the specified {chunk_size}"
                )
                # If single chunk is too long, split into more granular
                if len(current_doc) == 1:
                    split_again = split_into_granular_chunks(
                        current_doc[0], chunk_size, tokenizer
                    )
                    docs.extend(split_again)
                    current_doc = []
                    total = 0

            if len(current_doc) > 0:
                doc = separator.join(current_doc)
                if doc is not None:
                    docs.append(doc)
                while total > 0 or (
                    total + plen + (separator_len if len(current_doc) > 0 else 0)
                    > chunk_size
                    and total > 0
                ):
                    total -= get_prompt_length(
                        current_doc[0], tokenizer, no_special_tokens=True
                    ) + (separator_len if len(current_doc) > 1 else 0)
                    current_doc = current_doc[1:]

        current_doc.append(paragraph)
        total += plen + (separator_len if len(current_doc) > 1 else 0)
    # Check if the last one exceeds
    if (
        get_prompt_length(current_doc[-1], tokenizer, no_special_tokens=True)
        > chunk_size
        and len(current_doc) == 1
    ):
        split_again = split_into_granular_chunks(current_doc[0], chunk_size, tokenizer)
        docs.extend(split_again)
        current_doc = []
    else:
        doc = separator.join(current_doc)
        if doc is not None:
            docs.append(doc)

    return [doc for doc in docs if doc.strip()]


def split_sentences(text: str, spliter: str):
    """
    Splits text into sentences or segments based on a given delimiter while preserving punctuation.

    For punctuation-based splitters (e.g., ".", "!", "。"), it interleaves text and punctuation.
    For space-based splitting, it preserves trailing spaces.

    Args:
        text (str): The input text to split.
        spliter (str): Delimiter regex pattern (e.g., r"([.!?])", r"(。)", or " ").

    Returns:
        List[str]: List of split sentence-like segments with punctuation retained.
    """

    # Split by punctuation and keep punctuation
    text = text.strip()
    sentence_list = re.split(spliter, text)

    # Rearrange sentences and punctuation
    if spliter != " ":
        sentences = ["".join(i) for i in zip(sentence_list[0::2], sentence_list[1::2])]
        if len(sentence_list) % 2 != 0 and sentence_list[-1] != "":
            sentences.append(sentence_list[-1])
    else:
        sentences = [i + " " for i in sentence_list if i != ""]
        sentences[-1] = sentences[-1].strip()
    return sentences


def split_into_granular_chunks(
    text: str, chunk_size: int, tokenizer, spliter=r"([。！？；.?!;])",
) -> List[str]:
    """
    Splits long text into granular, token-length-constrained chunks using sentence boundaries.

    Sentences are first extracted using a delimiter pattern (`spliter`), then grouped into chunks such that
    each chunk does not exceed the specified `chunk_size` (in tokens). If a chunk still exceeds the limit,
    it is recursively broken down further using whitespace as a fallback.

    Ensures that the final chunks are balanced: if the last chunk is too small, it redistributes the last two
    chunks more evenly by re-splitting and re-allocating their sentences.

    Args:
        text (str): Input text to be chunked.
        chunk_size (int): Maximum number of tokens per chunk.
        tokenizer: Tokenizer instance with `.encode()` method to compute token length.
        spliter (str): Regex pattern to split sentences.

    Returns:
        List[str]: List of token-limited chunks, each composed of one or more sentences.
    """
    sentences = split_sentences(text, spliter)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence_length = get_prompt_length(sentence, tokenizer)

        if get_prompt_length(current_chunk, tokenizer) + sentence_length <= chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                if get_prompt_length(current_chunk, tokenizer) <= chunk_size:
                    chunks.append(current_chunk)
                else:
                    if spliter != " ":  # Avoid infinite loops
                        chunks.extend(
                            split_into_granular_chunks(
                                current_chunk,
                                chunk_size=chunk_size,
                                tokenizer=tokenizer,
                                spliter=" ",
                            )
                        )
            current_chunk = sentence

    if current_chunk != "":
        if get_prompt_length(current_chunk, tokenizer) <= chunk_size:
            chunks.append(current_chunk)
        else:
            if spliter != " ":  # Avoid infinite loops
                chunks.extend(
                    split_into_granular_chunks(
                        current_chunk,
                        chunk_size=chunk_size,
                        tokenizer=tokenizer,
                        spliter=" ",
                    )
                )

    # If last chunk too short, re-balance the last two chunks
    if len(chunks) > 1 and get_prompt_length(chunks[-1], tokenizer) < chunk_size // 2:
        last_chunk = chunks.pop()
        penultimate_chunk = chunks.pop()
        combined_text = penultimate_chunk + last_chunk

        new_sentences = split_sentences(combined_text, spliter)

        # Reallocate sentence using double pointer
        new_penultimate_chunk = ""
        new_last_chunk = ""
        start, end = 0, len(new_sentences) - 1

        while start <= end and len(new_sentences) != 1:
            flag = False
            if (
                get_prompt_length(
                    new_penultimate_chunk + new_sentences[start], tokenizer
                )
                <= chunk_size
            ):
                flag = True
                new_penultimate_chunk += new_sentences[start]
                if start == end:
                    break
                start += 1
            if (
                get_prompt_length(new_last_chunk + new_sentences[end], tokenizer)
                <= chunk_size
            ):
                new_last_chunk = new_sentences[end] + new_last_chunk
                end -= 1
                flag = True
            if not flag:
                break
        if start < end:
            # If there is any unallocated part, split it by punctuation or space and then allocate it
            remaining_sentences = new_sentences[start : end + 1]
            if remaining_sentences:
                remaining_text = "".join(remaining_sentences)
                words = remaining_text.split(" ")
                end_index = len(words) - 1
                for index, w in enumerate(words):
                    if (
                        get_prompt_length(
                            " ".join([new_penultimate_chunk, w]), tokenizer
                        )
                        <= chunk_size
                    ):
                        new_penultimate_chunk = " ".join([new_penultimate_chunk, w])
                    else:
                        end_index = index
                        break
                if end_index != len(words) - 1:
                    new_last_chunk = " ".join(words[end_index:]) + " " + new_last_chunk
        if len(new_sentences) == 1:
            chunks.append(penultimate_chunk)
            chunks.append(last_chunk)
        else:
            chunks.append(new_penultimate_chunk)
            chunks.append(new_last_chunk)

    return chunks
