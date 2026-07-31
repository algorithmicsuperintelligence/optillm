import re
import logging
import yaml
import json
import optillm
import time
import math_verify

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal, Any, Optional
from openai import BadRequestError as OpenAIBadRequestError
from openai import InternalServerError as OpenAIInternalServerError

logger = logging.getLogger(__name__)

@dataclass
class CepoConfig:
    bestofn_n: int  # number of responses to be generated in best of n stage
    bestofn_temperature: float  # temperature for verifier in best of n stage
    bestofn_max_tokens: int  # maximum number of tokens for verifier in best of n stage
    bestofn_rating_type: Literal["absolute", "pairwise", "majority"]  # type of rating in best of n stage
    planning_n: int  # number of plans generated in planning stage
    planning_m: int  # number of attempts to generate n plans in planning stage
    planning_temperature_step1: float  # temperature for generator in step 1 of planning stage
    planning_temperature_step2: float  # temperature for generator in step 2 of planning stage
    planning_temperature_direct_resp: float  # temperature for generator after step 2 if planning fails and answer directly
    planning_temperature_step3: float  # temperature for generator in step 3 of planning stage
    planning_temperature_step4: float  # temperature for generator in step 4 of planning stage
    planning_max_tokens_step1: int  # maximum number of tokens in step 1 of planning stage
    planning_max_tokens_step2: int  # maximum number of tokens in step 2 of planning stage
    planning_max_tokens_direct_resp: int  # maximum number of tokens after step 2 if planning fails and answer directly
    planning_max_tokens_step3: int  # maximum number of tokens in step 3 of planning stage
    planning_max_tokens_step4: int  # maximum number of tokens in step 4 of planning stage
    use_plan_diversity: bool  # whether to use plan diversity
    use_reasoning: bool # whether this model supports setting the reasoning effort parameter [True]
    use_reasoning_fallback: bool  # whether to fallback to lower levels of reasoning when higher level fails
    num_of_retries: int  # number of retries if llm call fails, 0 for no retries
    rating_model: Optional[str] = None # model to be used for rating
    print_output: bool = False  # whether to print the output of each stage


MCQ_PATTERNS = [
    # 0)"**Answer:** A" or "*Answers* – B", i.e. markdown‐wrapped "Answer(s)" with an unwrapped letter.
    re.compile(
        r'''(?ix)                   # case‐insensitive, ignore‐space
        (?:\*{1,2}|_{1,2})          # leading *…*  or _…_
        Answer[s]?                  #   Answer or Answers
        \s*[:\-–]?                  #   optional separator
        (?:\*{1,2}|_{1,2})          # closing wrapper
        \s*                         # optional space
        ([ABCD])\b                  # the actual letter
        ''',
        re.X
    ),

    # 0.1)
    re.compile(r'''(?ix)           # ignore case, allow verbose mode
        ^\s*                      # optional leading whitespace
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper
        Answer:?                   # the word 'answer' with an optional colon
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper again
        \s*:?\s*                  # optional colon with optional spaces
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper before letter
        ([ABCD])                 # capture the letter
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper after letter
        \s*                     # optional trailing whitespace, end of line
    ''', re.MULTILINE),

    # 1) Answer: (C)   or   Answers: (B)
    re.compile(r'(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*\(\s*([ABCD])\s*\)'),

    # 2) Answer: C    or   Answers – D
    re.compile(r'(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*([ABCD])\b'),

    # 3) Option B   or   Choice: C
    re.compile(r'(?ix)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCD])\b'),

    # 7) LaTeX \boxed{...A...}, catches both \boxed{A} and
    #    \boxed{\text{A } 2.08\times10^{-6}\,\mathrm{m}} etc.
    re.compile(r'(?x)\\boxed\{[^}]*?([ABCD])[^}]*\}', re.MULTILINE),

    # 7.5) LaTeX \boxed{\textbf{...C...}}
    re.compile(r'(?x)\\boxed\{[^}]*?\\textbf\{[^}]*?([ABCD])[^}]*\}[^}]*\}', re.MULTILINE),

    # 7.51) LaTeX \boxed{\text{...C...}}
    re.compile(r'(?x)\\boxed\{[^}]*?\\text\{[^}]*?([ABCD])[^}]*\}[^}]*\}', re.MULTILINE),

    # 4) bare singletons:  (A)  [B]
    re.compile(r'(?x)(?<![A-Za-z0-9])[\(\[]\s*([ABCD])\s*[\)\]](?![A-Za-z0-9])'),

    # 5) Markdown‐wrapped: *A*  **B**  _C_  __D__
    re.compile(r'(?x)(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCD])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])'),

    # 6) LaTeX \textbf{...C...}
    re.compile(r'(?x)\\textbf\{[^}]*?([ABCD])[^}]*\}'),

    # 8) markdown‐wrapped answer plus “)” plus description, e.g. **D) …**
    re.compile(r'''(?x)                        # ignore whitespace in pattern
        (?<![A-Za-z0-9])            # not preceded by word‐char
        (?:\*{1,2}|_{1,2})          # opening ** or __ or * or _
        \s*([ABCD])\)               # capture letter plus “)”
        [^*_\n]+?                   # some text inside wrapper
        (?:\*{1,2}|_{1,2})          # closing wrapper
        (?![A-Za-z0-9])             # not followed by word‐char
    '''),

    # 9) final fallback: a line that's exactly "A", "B.", "C)", "**D**", etc.
    re.compile(r'''(?x)^\s*
        (?:\*{1,2}|_{1,2})?     # optional markdown wrapper
        ([ABCD])                # capture group for letter
        (?:\*{1,2}|_{1,2})?     # optional closing markdown
        \s*[\.\)\-–:]?          # optional separator after the letter
        \s*.*$                  # allow any following text
    ''', re.MULTILINE),
]


# given command line arguments which includes a yaml file path, initialize a CePO configuration
def init_cepo_config(cmd_line_args: dict) -> CepoConfig:
    # get the command line arguments
    cepo_args = {
        key.split("cepo_")[1]: value
        for key, value in cmd_line_args.items()
        if "cepo" in key and "cepo_config_file" != key and value is not None
    }

    # get the yaml file arguments
    cepo_config_yaml = {}
    if cmd_line_args.get("cepo_config_file", None):
        with open(cmd_line_args["cepo_config_file"], "r") as yaml_file:
            cepo_config_yaml = yaml.safe_load(yaml_file)

    # merge cepo args from command line and yaml file, args from command line will overwrite the ones from yaml file
    cepo_args = {**cepo_config_yaml, **cepo_args}
    return CepoConfig(**cepo_args)


def extract_question_only(task: str) -> str:
    """
    We noticed that sometimes if the task includes specific formatting instructions, they may interfere with the reasoning flow. This
    is a temporary workaround to extract the question only from the task. Work in progress.
    """
    question_only = task.replace('\n## Question: \n\n', '')
    question_only = question_only.replace('\n\n\n## Instruction \n\nPlease answer this question by first reasoning and then providing your answer.\nPresent your reasoning and solution in the following json format. \nPlease show your final answer in the `answer` field, e.g.,`"answer": "42"`.\n\n```json\n{\n    "reasoning": "___",\n    "answer": "___"\n}\n```\n', '')
    return question_only


def remove_think_section(response):
    """
    Remove a <think>...</think> block from the response text, if present.

    Args:
        response (str): Raw model output.

    Returns:
        str: Response without the <think> section, or an empty string if input
        is invalid or empty.
    """
    if not isinstance(response, str) or not response:
        return ""
    if not response.startswith("<think>") and "<think>" not in response:
        return response
    match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    if match:
        parsed_response = match.group(1)
        return parsed_response
    else:
        return response


def extract_llm_response(response):
    """
    Extract text content and finish reason from an LLM response.

    Supports both non-streaming responses (dict-like with `.choices[0].message.content`)
    and streaming responses (iterable of chunks with `.choices[0].delta.content`).

    Args:
        response: LLM response object or streaming generator.

    Returns:
        Tuple[str, Optional[str]]:
            - Extracted text content (stripped).
            - Finish reason (or None if unavailable).
    """
    # Case 1: non-streaming response (dict-like object)
    if hasattr(response, "choices") and hasattr(response.choices[0], "message"):
        content = response.choices[0].message.content
        if content:
            content = content.strip()
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        return content, finish_reason

    # Case 2: streaming response (generator)
    full_content = ""
    finish_reason = None
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            full_content += delta.content
        if chunk.choices[0].finish_reason is not None:
            finish_reason = chunk.choices[0].finish_reason
    return full_content.strip(), finish_reason


def llm_call(
    client: Any,
    provider_request: dict,
    cepo_config: CepoConfig
) -> tuple[str, str, int]:
    """
    Call the LLM with retries on transient errors.

    Makes a chat completion request to the given client and extracts the response.
    Retries up to 2 times on 400/500 errors with exponential backoff.

    Args:
        client (Any): LLM API client instance.
        provider_request (dict): LMM call params.

    Returns:
        tuple[str, str, int]:
            - response_text: Model output (post-processed, never None).
            - finish_reason: Why generation stopped.
            - completion_tokens: Number of tokens generated.
    """
    retries = cepo_config.num_of_retries + 1  # total attempts = retries + 1 initial call
    for attempt in range(retries):
        try:
            response_object = client.chat.completions.create(
                stream=False,
                **provider_request
            )
            response_text, finish_reason = extract_llm_response(response_object)
            completion_tokens = getattr(getattr(response_object, "usage", None), "completion_tokens", 0) or 0
            response_text = response_text or ""  # Normalize None to ""
            if response_text is not None:
                response_text = remove_think_section(response_text)
            return response_text, finish_reason, completion_tokens

        except (OpenAIBadRequestError, OpenAIInternalServerError) as e:
            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.exception("Bad OpenAI Request")
            # Retry on 400 or 500
            if attempt < retries - 1:
                sleep_time = 0.2 * (attempt + 1)
                logger.debug(f"Got {e.__class__.__name__}, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue
            raise


def llm_call_reason_effort_fallback(
    client: Any,
    provider_request: dict,
    reasoning_effort_levels: list,
    cepo_config: CepoConfig
) -> tuple[Optional[Any], str, int]:
    """
    Call LLM with fallback on reasoning effort levels.

    This function wraps `llm_call` with retry and degradation logic to handle
    two main classes of errors:

    1. **Incomplete generation (finish_reason = "length")**:
       - The model returns a response object but does not finish generation
         (e.g., truncated output).
       - In this case, the reasoning effort is reduced, and another attempt
         is made with lower levels.

    2. **Server/validation errors (e.g., 400 BadRequest, 500 InternalServerError)**:
       - Often caused by gpt-oss's "expected output number" error, which cannot be
         fully recovered within the current API.
       - The function retries once, and if the error persists, reasoning effort
         is degraded to try again at lower levels.

    The fallback sequence continues until either:
      - A valid response is obtained (not truncated and not `None`), or
      - All reasoning effort levels are exhausted, in which case the last
        attempted result (possibly `None`) is returned.

    Args:
        client (Any): LLM API client instance used for making calls.
        provider_request (dict): LMM call params.
        reasoning_effort_levels (list): Ordered list of reasoning effort levels
            (e.g., ["high", "medium", "low"]) to try in fallback.

    Returns:
        tuple:
            - response: The LLM response object, or `None` if all attempts failed.
            - finish_reason (str): Reason why generation finished ("stop",
              "length", "error", etc.).
            - completion_tokens (int): Number of tokens generated in the final attempt.

    Notes:
        - This function prints diagnostic information when degrading reasoning effort.
        - For persistent server-side issues (400/500), degradation is attempted
          automatically, but a permanent fix may require upstream changes
          (see https://github.com/pydantic/pydantic-ai/issues/2449).
    """
    if not cepo_config.use_reasoning_fallback:
        reasoning_effort_levels = ["high"]
    for effort in reasoning_effort_levels:
        try:
            # Try with the current reasoning effort level
            if cepo_config.use_reasoning:
                provider_request["reasoning_effort"] = effort
            else:
                # Ensure reasoning_effort isn't set on the provider request
                provider_request.pop("reasoning_effort", None)
            response, finish_reason, completion_tokens = llm_call(
                client=client,
                provider_request=provider_request,
                cepo_config=cepo_config
            )
            if response is not None and finish_reason != "length":
                return response, finish_reason, completion_tokens
            logger.debug(f"Reasoning fallback from {effort}, to lower one")
        except OpenAIBadRequestError as bre:
            # Check to see if we attempted to use a reasoning effort not supported by the model
            if len(reasoning_effort_levels) == 1 and bre.message.startswith("Error code: 400 - {'error': {'message': 'think value"):
                logger.info(f"The think level {effort} was not supported by the model; Disabling thinking")
                cepo_config.use_reasoning = False
        except (OpenAIBadRequestError, OpenAIInternalServerError):
            # After 2 retries at this reasoning effort level it failed with error 400/500, lower level
            logger.debug(f"400/500 persisted after retries at reasoning effort {effort}; degrading effort")
            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.exception("Bad OpenAI Request; Fallback")
            continue

    return None, "error", 0


def generate_completion(system_prompt: str, task: str, client: Any, model: str, cepo_config: CepoConfig, approach: Optional[str] = None, request_id: str = None) -> str:
    """
    Generates a completion based on the provided system prompt and task.

    Parameters:
        system_prompt (str): The system prompt to guide the model.
        task (str): The task or question to be addressed.
        client (Any): The client instance for interacting with the AI model.
        model (str): The model name to be used for generating completions.
        cepo_config (CepoConfig): Configuration parameters for CePO flow.
        approach (str|None): optional approach that is used to seed plan generation.

    Returns:
        Tuple[str, int, dict]: The generated completion, number of tokens used, and a log dictionary.
    """
    completion_tokens = 0
    question_only = extract_question_only(task)
    cb_log = {}
    plans = []

    def generate_single_plan(i):
        local_cb_log = {}
        local_completion_tokens = 0

        if cepo_config.planning_max_tokens_step1 != 0:
            if cepo_config.use_plan_diversity:
                assert approach
                content = (
                    f"To answer this question, can you come up with a concise plan using to solve it step-by-step but do not provide the "
                    f"final answer. Here is the approach you need to follow to generate the plan: {approach}. "
                    f"Also, for each step, provide your confidence in the correctness of that step as well as your ability "
                    f"to execute it correctly. Here is the question:\n{question_only}\nRead the question again:\n\n{question_only}"
                )
            else:
                assert not approach
                content = (
                    f"To answer this question, can you come up with a concise plan to solve it step-by-step but do not provide the "
                    f"final answer. Also, for each step, provide your confidence in the correctness of that step as well as your ability "
                    f"to execute it correctly. Here is the question:\n{question_only}\nRead the question again:\n\n{question_only}"
                )

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

            provider_request = {
                "model": model,
                "messages": messages,
                "max_tokens": cepo_config.planning_max_tokens_step1,
                "temperature": cepo_config.planning_temperature_step1,
                "top_p": 1.0
            }

            response, finish_reason, completion_tokens = llm_call_reason_effort_fallback(
                client=client,
                provider_request=provider_request,
                reasoning_effort_levels=["high", "medium"],
                cepo_config=cepo_config
            )
            local_completion_tokens += completion_tokens
            # Log provider call if conversation logging is enabled
            if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)

            if finish_reason == "length":
                return i, None, local_completion_tokens, {f"messages_planning_{i}_rejected_due_to_length": messages}
            parsed_plan = response
        else:
            messages = []
            parsed_plan = ""

        # Step 2 – Execute plan
        if cepo_config.planning_max_tokens_step1 != 0:
            messages.append({"role": "assistant", "content": parsed_plan})
            messages.append({"role": "user", "content": "Can you execute the above plan step-by-step to produce the final answer. Be extra careful when executing steps where your confidence is lower. /think"})
        else:
            messages.append({"role": "user", "content": f"Can you solve this problem step-by-step to produce the final answer? Here is the question:\n{question_only}\nRead the question again:\n\n{question_only} /think"})

        provider_request = {
                "model": model,
                "messages": messages,
                "max_tokens": cepo_config.planning_max_tokens_step1,
                "temperature": cepo_config.planning_temperature_step1,
                "top_p": 1.0
                }

        response, finish_reason, completion_tokens = llm_call_reason_effort_fallback(
                client=client,
                provider_request=provider_request,
                reasoning_effort_levels=["high", "medium"],
                cepo_config=cepo_config
            )
        local_completion_tokens += completion_tokens

        # Log provider call if conversation logging is enabled
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        if finish_reason == "length":
            return i, None, local_completion_tokens, {f"messages_planning_{i}_rejected_due_to_length": messages}

        parsed_exec = response
        messages.append({"role": "assistant", "content": parsed_exec})
        local_cb_log[f"messages_planning_{i}"] = messages
        return i, parsed_exec, local_completion_tokens, local_cb_log

    # Step 1 & 2: Parallel planning + execution
    with ThreadPoolExecutor(max_workers=cepo_config.planning_m) as executor:
        futures = [executor.submit(generate_single_plan, i) for i in range(cepo_config.planning_m)]

        for future in as_completed(futures):
            i, plan, tokens_used, log_entry = future.result()
            completion_tokens += tokens_used
            cb_log.update(log_entry)
            if plan:
                plans.append((i, plan))
                if cepo_config.print_output:
                    print(f"\nCePO: Plan proposal generated. Attempt {i + 1} out of {cepo_config.planning_m}.\n")
            if len(plans) == cepo_config.planning_n:
                break

    plans = [plan for _, plan in sorted(plans)]  # keep original order

    if not plans:
        # If no plans were generated, attempt to answer directly
        messages = [
            {"role": "user", "content": question_only},
        ]

        provider_request = {
            "model": model,
            "messages": messages,
            "max_tokens": cepo_config.planning_max_tokens_direct_resp,
            "temperature":cepo_config.planning_temperature_direct_resp,
            "top_p": 0.95,
        }

        response, finish_reason, tokens_used = llm_call_reason_effort_fallback(
                    client=client,
                    provider_request=provider_request,
                    reasoning_effort_levels=["high", "medium", "low"],
                    cepo_config=cepo_config
                )
        completion_tokens += tokens_used

        # Log provider call if conversation logging is enabled
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        if response is None or finish_reason == "length":
            print("Direct answer failed, empty response or length")
            response = ""
        messages.append({"role": "assistant", "content": response})

        plans.append(response)
        cb_log["messages_planning_fallback_used"] = messages
        if cepo_config.print_output:
            print("\nCePO: No plans generated successfully. Taking the fallback.\n")

    # Step 3 - Review and consolidate plans
    plans_message = ""
    for i, plan in enumerate(plans):
        plans_message += f"Response {i + 1}:\n{plan}\n\n"
    plans_message = plans_message.rstrip()

    content = f"Can you review your last {len(plans)} responses and identify any inconsistency between them. After that, can you address "\
              f"it and present a final step-by-step solution to the problem? Here is the question:\n{question_only} /think"

    user_content = f"Previous responses to review:\n\n{plans_message}\n\n{content}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    provider_request = {
                "model": model,
                "messages": messages,
                "max_tokens": cepo_config.planning_max_tokens_step1,
                "temperature": cepo_config.planning_temperature_step1,
                "top_p": 1.0
                }

    response, finish_reason, tokens_used = llm_call_reason_effort_fallback(
                client=client,
                provider_request=provider_request,
                reasoning_effort_levels=["high", "medium"],
                cepo_config=cepo_config
            )
    completion_tokens += tokens_used

    # Log provider call if conversation logging is enabled
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)

    if response is None or finish_reason == "length":
        print("Step 3 failed and only taking plans[0]")
        final_solution = plans[0]
    else:
        #completion_tokens += completion_tokens # TODO: This seems like a bug
        final_solution = response
    messages.append({"role": "assistant", "content": final_solution})

    # Step 4 – Final answer
    if cepo_config.planning_max_tokens_step4 != 0:
        content = f"Use your final solution from above to correctly answer the question. Here is the question:\n{task} /think"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here's my final solution: {final_solution}\n\nNow {content}"}
        ]

        provider_request = {
                "model": model,
                "messages": messages,
                "max_tokens": cepo_config.planning_max_tokens_step1,
                "temperature": cepo_config.planning_temperature_step1,
                "top_p": 1.0
                }

        response, finish_reason, tokens_used = llm_call_reason_effort_fallback(
                client=client,
                provider_request=provider_request,
                reasoning_effort_levels=["high", "medium"],
                cepo_config=cepo_config
            )
        completion_tokens += tokens_used

        # Log provider call if conversation logging is enabled
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        if response is None or finish_reason == "length":
            print("Step 4 failed and only taking step 3 output")
            final_output = final_solution
        else:
            final_output = response
    else:
        final_output = final_solution

    cb_log["messages"] = messages
    if cepo_config.print_output:
        print("\nCePO: Answer generated for one bestofn_n attempt.")

    return final_output, completion_tokens, cb_log


def generate_approaches(system_prompt: str, initial_query: str, num_approach: int, client: Any, model: str, cepo_config: CepoConfig, max_retry: int = 2, request_id: str = None) -> tuple[list[str], int]:
    completion_tokens = 0
    question_only = extract_question_only(initial_query)
    approaches = []
    content = f'To answer the question: "{question_only}", please propose {num_approach} different high-level approaches to solve the problem. '\
              f'All approaches should be fundamentally different from each other and easily excecutable without too much steps. Do not include a '\
              f'step-by-step plan or the final answer. You must present the approaches in the following JSON format which is directly loadable:\n'\
              f'{{\n'\
              f'    "approach_1": "<Description of approach 1>",\n'\
              f'    "approach_2": "<Description of approach 2>",\n'\
              f'    "approach_3": "<Description of approach 3>",\n'\
              f'    ...\n'\
              f'}}'
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

    retries = 0
    while retries < max_retry:
        try:
            # Prepare request for logging
            provider_request = {
                "model": model,
                "messages": messages,
                "max_tokens": cepo_config.planning_max_tokens_step0,
                "temperature": cepo_config.planning_temperature_step0,
                "stream": False,
            }

            response = client.chat.completions.create(**provider_request)

            # Log provider call if conversation logging is enabled
            if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
            completion_tokens += response.usage.completion_tokens
            completion = response.choices[0].message.content

            # Try to parse the completion as JSON, escape latex math symbols
            cleaned_completion = completion.replace('\\', '\\\\').replace('json','').replace("```", "")
            for _, value in json.loads(cleaned_completion).items():
                approaches.append(value.replace('\\\\', '\\'))
            break  # Exit the loop if parsing is successful

        except json.JSONDecodeError as e:
            # If there's an error, print a message and regenerate the content
            print(e)
            print(f"Parsing Error when generating diverse approaches, retrying... ({retries + 1}/{max_retry})")

            retries += 1

    if retries == max_retry:
        print("Max retry attempts reached, returning empty list.")
        return [], 0  # Default approach

    return approaches, completion_tokens


def generate_n_completions(system_prompt: str, initial_query: str, client: Any, model: str, cepo_config: CepoConfig, request_id: str) -> tuple[list[str], int, dict]:
    """
    Generates n completions for the Best of N step of CePO.

    Parameters:
        system_prompt (str): The system prompt to guide the model.
        initial_query (str): The task or question to be addressed.
        client (Any): The client instance for interacting with the AI model.
        model (str): The model name to be used for generating completions.
        cepo_config (CepoConfig): Configuration parameters for CePO flow.

    Returns:
        Tuple[str, int, dict]: The generated completion, number of tokens used, and a log dictionary.
    """
    completion_tokens = 0
    cb_log = {}
    cb_log["system_prompt"] = system_prompt
    cb_log["initial_query"] = initial_query
    completions = [None] * cepo_config.bestofn_n
    approaches = None

    # Generate Approach and Descriptions
    if cepo_config.use_plan_diversity:
        approaches, approach_completion_tokens = generate_approaches(
            system_prompt=system_prompt,
            initial_query=initial_query,
            num_approach=cepo_config.bestofn_n,
            client=client,
            model=model,
            cepo_config=cepo_config,
            request_id=request_id
        )
        cb_log["approaches"] = approaches
        completion_tokens += approach_completion_tokens
        if cepo_config.print_output or logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug(f"\nCePO: Plan diversity approaches ({cepo_config.bestofn_n}):\n{approaches}\n")

    def run_single_completion(i):
        if cepo_config.print_output or logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug(f"\nCePO: Generating completion {i + 1} out of {cepo_config.bestofn_n} \n")
        approach = approaches[i] if approaches else None
        response_i, completion_tokens_i, cb_log_i = generate_completion(system_prompt, initial_query, client, model, cepo_config, approach, request_id)
        return i, response_i, completion_tokens_i, cb_log_i

    # Run in parallel
    with ThreadPoolExecutor(max_workers=cepo_config.bestofn_n) as executor:
        futures = [executor.submit(run_single_completion, i) for i in range(cepo_config.bestofn_n)]
        for future in as_completed(futures):
            i, response_i, tokens_i, cb_log_i = future.result()
            completions[i] = response_i
            completion_tokens += tokens_i
            cb_log[f"completion_{i}_response"] = response_i
            cb_log[f"completion_{i}_log"] = cb_log_i
            cb_log[f"completion_{i}_completion_tokens"] = tokens_i

    if cepo_config.print_output or logger.getEffectiveLevel() == logging.DEBUG:
        logger.debug("\nCePO: All Answers generated!")

    completions = [c if isinstance(c, str) else "" for c in completions]
    return completions, completion_tokens, cb_log


def rate_completions_absolute(system_prompt: str, initial_query: str, client: Any, model: str, completions: list[str], cepo_config: CepoConfig, cb_log: dict, request_id: str = None) -> tuple[str, int, dict]:
    """
    Rates completions for the Best of N step of CePO. Each completion is rated on a scale of 1 to 10 individually.

    Parameters:
        system_prompt (str): The system prompt to guide the model.
        initial_query (str): The task or question to be addressed.
        client (Any): The client instance for interacting with the AI model.
        model (str): The model name to be used for generating completions.
        completions (list[str]): List of completions to be rated.
        cepo_config (CepoConfig): Configuration parameters for CePO flow.

    Returns:
        Tuple[str, int, dict]: The generated completion, number of tokens used, and a log dictionary.
    """
    completion_tokens = 0
    rating_prompt = "Please act as an impartial judge and evaluate the accuracy of the response provided by an AI assistant to "\
              "the user question displayed below. Your evaluation should consider only correctness and accuracy as the primary factor. "\
              "Evaluation Criteria:\n"\
              "- Correctness: How free is it from errors or mistakes?\n"\
              "- Accuracy: Are the information and explanations factually correct?\n"\
              "Evaluation Process:\n"\
              "1. Carefully review the user question and the AI assistant's response.\n"\
              "2. Assess the response for any inaccuracies in reasoning as well as execution.\n"\
              "3. Provide a detailed explanation of your step-by-step evaluation.\n"\
              "4. Identify if the final answer is correct or not. \n"\
              "Begin your evaluation by thinking through the given problem and response step-by-step. "\
              "VERY IMPORTANT: Re-do any calculations present and check if you arrive at the same answer. "\
              "Throughly check for any inaccuracies in reasoning and calculations for each step. "\
              "Be as objective as possible. After providing your detailed explanation, "\
              "please rate the response as 0 or 1, (0 for incorrect and 1 for correct) by strictly following this format: "\
              "\"Rating: [[rating]]\", for example: \"Rating: [[0]]\""

    rating_format_instruction = "\n\nRate the above response beginning with the detailed explanation followed by a rating of 0 or 1 by strictly following this format: \"Explanation: <reason for your rating>\n\nRating: [[rating]]\""

    ratings = []
    for i, completion in enumerate(completions):
        # Create a fresh conversation with proper role alternation for each completion
        system_content = f"USER QUESTION: {initial_query}\n\nRESPONSE: {completion}"
        rating_messages = [
            {"role": "system", "content": system_prompt + "\n\n" + rating_prompt},
            {"role": "user", "content": system_content + rating_format_instruction}
        ]

        # Prepare request for logging
        provider_request = {
            "model": model,
            "messages": rating_messages,
            "max_tokens": cepo_config.bestofn_max_tokens,
            "temperature": cepo_config.bestofn_temperature,
            "top_p": 1.0
        }

        rating_response = client.chat.completions.create(**provider_request)
        rating_response, _, completion_tokens_ = llm_call_reason_effort_fallback(
                client=client,
                provider_request=provider_request,
                reasoning_effort_levels=["high", "medium"],
                cepo_config=cepo_config
            )

        # Log provider call if conversation logging is enabled
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
            response_dict = rating_response.model_dump() if hasattr(rating_response, 'model_dump') else rating_response
            optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        completion_tokens += completion_tokens_

        cb_log[f"rating_response_{i}"] = rating_response
        if cepo_config.print_output:
            print(f"\nCePO: Rating response for completion {i}: {rating_response}")

        pattern = r"Rating: \[\[(\d+)\]\]"
        match = re.search(pattern, rating_response)
        rating_response = match.group(1) if match else "-1"  # parsing error results in a rating of -1

        try:
            ratings.append(float(rating_response))
        except ValueError:
            ratings.append(-1)

    best_index = ratings.index(max(ratings))
    cb_log["ratings"] = ratings
    cb_log["best_index"] = best_index
    if cepo_config.print_output:
        print(f"\nCePO: Finished rating completions. Ratings: {ratings}, best completion index: {best_index}")
    return completions[best_index], completion_tokens, cb_log


def rate_completions_pairwise(system_prompt: str, initial_query: str, client: Any, model: str, completions: list[str], cepo_config: CepoConfig, cb_log: dict, request_id: str = None) -> tuple[str, int, dict]:
    """
    Rates completions for the Best of N step of CePO. Completions are rated pairwise against each other in both orders (A vs B and B vs A).

    Parameters:
        system_prompt (str): The system prompt to guide the model.
        initial_query (str): The task or question to be addressed.
        client (Any): The client instance for interacting with the AI model.
        model (str): The model name to be used for generating completions.
        completions (list[str]): List of completions to be rated.
        cepo_config (CepoConfig): Configuration parameters for CePO flow.

    Returns:
        Tuple[str, int, dict]: The generated completion, number of tokens used, and a log dictionary.
    """
    completion_tokens = 0
    rating_prompt = "Please act as an impartial judge and compare the quality of the two responses provided by the AI assistant " \
              "to the user's question displayed below. Evaluation Criteria:\n" \
              "- Helpfulness: How effectively does the response meet the user's needs?\n" \
              "- Relevance: How directly does the response address the original question?\n" \
              "- Accuracy: Are the information and explanations factually correct?\n" \
              "- Depth: Does the response provide comprehensive and meaningful insights?\n" \
              "- Creativity: Does the response offer unique or innovative perspectives?\n" \
              "- Clarity: Is the response well-organized, coherent, and easy to understand?\n" \
              "Evaluation Process:\n" \
              "1. Carefully review the user's question and the AI assistant's responses.\n" \
              "2. Compare the responses against each other for each criterion.\n" \
              "3. Provide a concise explanation of your overall evaluation.\n" \
              "4. Select the response that is superior based on the above criteria.\n" \
              "Reply with \"Better Response: [[response id]]\".\n" \
              "If the first response is better, reply with \"Better Response: [[0]]\". " \
              "If the second response is better, reply with \"Better Response: [[1]]\"."

    ratings = [0] * cepo_config.bestofn_n
    pairs = [(i, j) for i in range(cepo_config.bestofn_n) for j in range(cepo_config.bestofn_n) if i != j]

    for pair in pairs:
        # Create comparison content with both responses
        comparison_content = f"User Question: {initial_query}\n\n" \
                           f"Response 0: {completions[pair[0]]}\n\n" \
                           f"Response 1: {completions[pair[1]]}\n\n" \
                           f"Which response is better? Please provide your reasoning and then indicate your choice with \"Better Response: [[0]]\" if the first response is better, or \"Better Response: [[1]]\" if the second response is better."

        # Create a fresh conversation for each comparison with proper system→user structure
        rating_messages = [
            {"role": "system", "content": system_prompt + "\n\n" + rating_prompt},
            {"role": "user", "content": comparison_content}
        ]

        # Prepare request for logging
        provider_request = {
            "model": model,
            "messages": rating_messages,
            "max_tokens": cepo_config.bestofn_max_tokens,
            "temperature": cepo_config.bestofn_temperature
        }
        rating_response = client.chat.completions.create(**provider_request)

        # Log provider call if conversation logging is enabled
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
            response_dict = rating_response.model_dump() if hasattr(rating_response, 'model_dump') else rating_response
            optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        completion_tokens += rating_response.usage.completion_tokens
        rating_response = rating_response.choices[0].message.content.strip()

        cb_log[f"rating_response_for_pair_{pair[0]}_{pair[1]}"] = rating_response
        if cepo_config.print_output:
            print(f"\nCePO: Rating response for pair {pair}: {rating_response}")

        pattern = r"Better Response: \[\[(\d+)\]\]"
        match = re.search(pattern, rating_response)
        if match:
            rating_response = match.group(1)
            try:
                rating = int(rating_response)
                ratings[pair[rating]] += 1
            except ValueError:
                ratings[pair[0]] += 1  # if parsing unsuccessful, default to the first response
        else:
            ratings[pair[0]] += 1  # if parsing unsuccessful, default to the first response

    best_index = ratings.index(max(ratings))
    cb_log["ratings"] = ratings
    cb_log["best_index"] = best_index
    if cepo_config.print_output:
        print(f"\nCePO: Finished rating completions. Ratings: {ratings}, best completion index: {best_index}")
    return completions[best_index], completion_tokens, cb_log


def extract_answer_mathverify(response_str, last_n_chars=100):
    response_str = str(response_str)
    try:
        float(response_str)
        return [float(response_str)]
    except Exception:
        response_str = response_str.split("</think>", 1)[1] if "</think>" in response_str else response_str
        if last_n_chars is not None:
            response_str = response_str[-last_n_chars:]
        parsed_result = math_verify.parse(response_str, parsing_timeout=None)
        return parsed_result


def extract_abcd(text: str) -> str | None:
    """
    Scan text (with Markdown/LaTeX wrappers intact) and return
    'A', 'B', 'C', or 'D' if a correct-answer declaration is found.
    Otherwise return None.
    """
    matches = []
    for prio, pat in enumerate(MCQ_PATTERNS):
        m = pat.search(text)
        if m:
            letter = m.group(1).upper()
            if letter in 'ABCD':
                matches.append((prio, m, letter))

    matches.sort(key=lambda triple: (
        triple[0],
        len(triple[1].group(0))
    ))
    for _, match, letter in matches:
        return letter
    return text.removeprefix('**')[:1]


def majority_vote_math(completions, last_n_chars=100):
    extracted_answer_map = []
    for response in completions:
        extracted_answer = extract_answer_mathverify(response, last_n_chars)
        extracted_answer = extracted_answer[0] if extracted_answer else None
        extracted_answer_map.append((response, extracted_answer))

    counts = Counter(answer for _, answer in extracted_answer_map)
    common_list = counts.most_common(1)
    majority_answer, count = common_list and common_list[0] or (None, 0)

    for response, answer in extracted_answer_map:
        if answer == majority_answer:
            return response, count
    if extracted_answer_map and extracted_answer_map[0]:
        return extracted_answer_map[0][0], 0
    return None, 0


def majority_vote_mcq(completions, last_n_chars=100):
    extracted_answer_map = []
    for response in completions:
        extracted_answer = extract_abcd(response[-last_n_chars:])
        extracted_answer_map.append((response, extracted_answer))

    counts = Counter(answer for _, answer in extracted_answer_map)
    common_list = counts.most_common(1)
    majority_answer, count = common_list and common_list[0] or (None, 0)

    for response, answer in extracted_answer_map:
        if answer == majority_answer:
            return response, count
    if extracted_answer_map and extracted_answer_map[0]:
        return extracted_answer_map[0][0], 0
    return None, 0


def rate_completions_majority(completions: list[str], last_n_chars: int = 150) -> tuple[str, int, dict]:
    mcq_majority, count = majority_vote_mcq(completions, last_n_chars)
    if mcq_majority is None:
        return majority_vote_math(completions, last_n_chars)
    return mcq_majority, count


def cepo(system_prompt: str, initial_query: str, client: Any, model: str, cepo_config: CepoConfig, request_id: str = None) -> tuple[str, int]:
    """
    Applies CePO reasoning flow for the given task. First, it generates multiple completions, and then rates them to select the best one.
    Each completion is generated as follows:

    Generate `planning_n` solution proposals:
        Step 1: Plan Generation - The model generates a detailed, step-by-step plan to solve the problem, along with its confidence level for
                each step.
        Step 2: Initial Solution - Using the plan from Step 1, the model produces an initial solution.

    Step 3: Plan Refinement - The model reviews all generated solution proposals and their associated plans, identifying inconsistencies.
            Based on this analysis, a refined, final step-by-step plan is constructed.
    Step 4: Final Solution - The model uses the refined plan from Step 3 to produce the final answer.

    Parameters:
        system_prompt (str): The system prompt to guide the model.
        initial_query (str): The task or question to be addressed.
        client (Any): The client instance for interacting with the AI model.
        model (str): The model name to be used for generating completions.
        cepo_config (CepoConfig): Configuration parameters for CePO flow.

    Returns:
        Tuple[str, int, dict]: The generated completion, number of tokens used
    """

    logger.info("CePO Started; Generating completions")

    # Generate completions
    completions, completion_tokens_planning, cb_log = generate_n_completions(system_prompt, initial_query, client, model, cepo_config, request_id)  # cb_log is a dictionary for debugging purposes
    completions = [c for c in completions if c]  # safeguard in case completion is None (observed with GPT OSS)

    # Rate the completions
    rating_model = cepo_config.rating_model if cepo_config.rating_model else model
    if cepo_config.bestofn_rating_type == "absolute":
        best_completion, completion_tokens_rating, cb_log = rate_completions_absolute(system_prompt, initial_query, client, rating_model, completions, cepo_config, cb_log, request_id)
    elif cepo_config.bestofn_rating_type == "pairwise":
        best_completion, completion_tokens_rating, cb_log = rate_completions_pairwise(system_prompt, initial_query, client, rating_model, completions, cepo_config, cb_log, request_id)
    elif cepo_config.bestofn_rating_type == "majority":
        best_completion, _ = rate_completions_majority(completions)
        completion_tokens_rating = 0
    else:
        raise ValueError("Invalid rating type in cepo_config")

    return best_completion, completion_tokens_planning + completion_tokens_rating
