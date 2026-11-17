"""
Chain of Code (CoC) plugin for OptILLM.

This plugin implements a chain-of-code approach that combines Chain-of-Thought (CoT)
reasoning with code execution and LLM-based code simulation.

SAFETY NOTE: This plugin has been refactored to use Jupyter notebook kernel execution
instead of direct exec() calls. This provides process isolation and prevents potentially
dangerous code from crashing or affecting the main OptILLM process.

Key safety improvements:
- Code runs in isolated notebook kernels (separate processes)
- 30-second timeout prevents infinite loops
- Main process is protected from crashes, system exits, and memory issues
- Matplotlib/visualization code is safely removed to prevent display issues
- Comprehensive error handling and recovery
"""

import re
import logging
from typing import Tuple, Any, List
import ast
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import tempfile

logger = logging.getLogger(__name__)

# Plugin identifier
SLUG = "coc"

# Maximum attempts to fix code
MAX_FIX_ATTEMPTS = 3

# Initial code generation prompt
CHAIN_OF_CODE_PROMPT = """
Write Python code to solve this problem. The code should:
1. Break down the problem into clear computational steps
2. Use standard Python features and math operations
3. Store the final result in a variable named 'answer'
4. Include error handling where appropriate
5. Be complete and executable

Format your response using:
```python
[Your complete Python program here]
```
"""

# Code fix prompt
CODE_FIX_PROMPT = """
The following Python code failed to execute. Fix the code to make it work.
Original code:
```python
{code}
```

Error encountered:
{error}

Please provide a complete, fixed version of the code that:
1. Addresses the error message
2. Maintains the same logic and approach
3. Stores the final result in 'answer'
4. Is complete and executable

Return only the fixed code in a code block:
```python
[Your fixed code here]
```
"""

# Simulation prompt
SIMULATION_PROMPT = """
The following Python code could not be executed directly. Analyze the code and determine what the answer would be.
Pay special attention to:
1. The core computational logic, ignoring any visualization or display code
2. The key mathematical operations that determine the final answer
3. Any logic that affects the 'answer' variable

Code to analyze:
```python
{code}
```

Runtime error encountered:
{error}

Return ONLY the final value that would be in the 'answer' variable. Return just the value, no explanations.
"""


def extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text."""
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    blocks = [m.strip() for m in matches]
    logger.info(f"Extracted {len(blocks)} code blocks")
    for i, block in enumerate(blocks):
        logger.info(f"Code block {i+1}:\n{block}")
    return blocks


def sanitize_code(code: str) -> str:
    """Prepare code for safe execution by removing problematic visualization code."""
    # Remove or modify problematic visualization code
    lines = code.split("\n")
    safe_lines = []
    for line in lines:
        # Skip matplotlib-related imports and plotting commands that could cause issues
        if any(
            x in line.lower()
            for x in ["matplotlib", "plt.", ".plot(", ".show(", "figure", "subplot"]
        ):
            # Replace with a comment to maintain code structure
            safe_lines.append(f"# {line}  # Removed for safety")
        else:
            # Keep the line if it's not visualization-related
            safe_lines.append(line)

    return "\n".join(safe_lines)


def execute_code(code: str) -> Tuple[Any, str]:
    """Attempt to execute the code using Jupyter notebook kernel and return result or error."""
    logger.info("Attempting to execute code in notebook kernel")
    logger.info(f"Code:\n{code}")

    try:
        # Sanitize the code first
        sanitized_code = sanitize_code(code)

        # Create a notebook with the code
        notebook = nbformat.v4.new_notebook()

        # Add code that captures the answer variable
        enhanced_code = f"""
{sanitized_code}

# Capture the answer variable for output
if 'answer' in locals():
    print(f"ANSWER_RESULT: {{answer}}")
else:
    print("ANSWER_RESULT: No answer variable found")
"""

        notebook["cells"] = [nbformat.v4.new_code_cell(enhanced_code)]

        # Convert notebook to JSON string and then to bytes
        notebook_json = nbformat.writes(notebook)
        notebook_bytes = notebook_json.encode("utf-8")

        # Create temporary notebook file
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".ipynb", delete=False
        ) as tmp:
            tmp.write(notebook_bytes)
            tmp.flush()
            tmp_name = tmp.name

        try:
            # Read and execute the notebook
            with open(tmp_name, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Execute with timeout and isolation
            ep = ExecutePreprocessor(timeout=30, kernel_name="python3")
            ep.preprocess(nb, {"metadata": {"path": "./"}})

            # Extract the output
            output = ""
            error_output = ""

            for cell in nb.cells:
                if cell.cell_type == "code" and cell.outputs:
                    for output_item in cell.outputs:
                        if output_item.output_type == "stream":
                            if output_item.name == "stdout":
                                output += output_item.text
                            elif output_item.name == "stderr":
                                error_output += output_item.text
                        elif output_item.output_type == "execute_result":
                            output += str(output_item.data.get("text/plain", ""))
                        elif output_item.output_type == "error":
                            error_output += f"{output_item.ename}: {output_item.evalue}"

            # Check for errors first
            if error_output:
                logger.error(f"Execution failed: {error_output}")
                return None, error_output

            # Parse the answer from output
            output = output.strip()

            # Look for our special ANSWER_RESULT marker
            if "ANSWER_RESULT:" in output:
                answer_line = [
                    line for line in output.split("\n") if "ANSWER_RESULT:" in line
                ][-1]
                answer_str = answer_line.split("ANSWER_RESULT:", 1)[1].strip()

                if answer_str == "No answer variable found":
                    error = "Code executed but did not produce an answer variable"
                    logger.warning(error)
                    return None, error

                try:
                    # Try to evaluate the answer to convert it to proper type
                    answer = ast.literal_eval(answer_str)
                except (ValueError, SyntaxError):
                    # If literal_eval fails, keep as string
                    answer = answer_str

                logger.info(f"Execution successful. Answer: {answer}")
                return answer, None
            else:
                # Fallback: try to extract answer from any output
                if output:
                    logger.info(f"Execution completed with output: {output}")
                    return output, None
                else:
                    error = "Code executed but produced no output"
                    logger.warning(error)
                    return None, error

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_name)
            except:
                pass

    except Exception as e:
        error = f"Notebook execution failed: {str(e)}"
        logger.error(error)
        return None, error


def generate_fixed_code(
    original_code: str, error: str, client, model: str
) -> Tuple[str, int]:
    """Ask LLM to fix the broken code."""
    logger.info("Requesting code fix from LLM")
    logger.info(f"Original error: {error}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": CODE_FIX_PROMPT.format(code=original_code, error=error),
            },
            {"role": "user", "content": "Fix the code to make it work."},
        ],
        temperature=0.2,
    )

    if not response.choices:
        raise ValueError("LLM returned empty response in generate_fixed_code")
    fixed_code = response.choices[0].message.content
    code_blocks = extract_code_blocks(fixed_code)

    if code_blocks:
        logger.info("Received fixed code from LLM")
        return code_blocks[0], response.usage.completion_tokens
    else:
        logger.warning("No code block found in LLM response")
        return None, response.usage.completion_tokens


def simulate_execution(code: str, error: str, client, model: str) -> Tuple[Any, int]:
    """Ask LLM to simulate code execution."""
    logger.info("Attempting code simulation with LLM")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SIMULATION_PROMPT.format(code=code, error=error),
            },
            {
                "role": "user",
                "content": "Simulate this code and return the final answer value.",
            },
        ],
        temperature=0.2,
    )

    try:
        if not response.choices:
            raise ValueError("LLM returned empty response in simulate_execution")
        result = response.choices[0].message.content.strip()
        # Try to convert to appropriate type
        try:
            answer = ast.literal_eval(result)
        except:
            answer = result
        logger.info(f"Simulation successful. Result: {answer}")
        return answer, response.usage.completion_tokens
    except Exception as e:
        logger.error(f"Failed to parse simulation result: {str(e)}")
        return None, response.usage.completion_tokens


def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """Main Chain of Code execution function."""
    logger.info("Starting Chain of Code execution")
    logger.info(f"Query: {initial_query}")

    # Initial code generation
    messages = [
        {"role": "system", "content": system_prompt + "\n" + CHAIN_OF_CODE_PROMPT},
        {"role": "user", "content": initial_query},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7
    )
    total_tokens = response.usage.completion_tokens

    # Extract initial code
    if not response.choices:
        raise ValueError("LLM returned empty response in run (initial code generation)")
    code_blocks = extract_code_blocks(response.choices[0].message.content)
    if not code_blocks:
        logger.warning("No code blocks found in response")
        return response.choices[0].message.content, total_tokens

    current_code = code_blocks[0]
    fix_attempts = 0
    last_error = None

    # Strategy 1: Direct execution and fix attempts
    while fix_attempts < MAX_FIX_ATTEMPTS:
        fix_attempts += 1
        logger.info(f"Execution attempt {fix_attempts}/{MAX_FIX_ATTEMPTS}")

        # Try to execute current code
        answer, error = execute_code(current_code)

        # If successful, return the answer
        if error is None:
            logger.info(f"Successful execution on attempt {fix_attempts}")
            return str(answer), total_tokens

        last_error = error

        # If we hit max attempts, break to try simulation
        if fix_attempts >= MAX_FIX_ATTEMPTS:
            logger.warning(f"Failed after {fix_attempts} fix attempts")
            break

        # Otherwise, try to get fixed code from LLM
        logger.info(f"Requesting code fix, attempt {fix_attempts}")
        fixed_code, fix_tokens = generate_fixed_code(current_code, error, client, model)
        total_tokens += fix_tokens

        if fixed_code:
            current_code = fixed_code
        else:
            logger.error("Failed to get fixed code from LLM")
            break

    # Strategy 2: If all execution attempts failed, try simulation
    logger.info("All execution attempts failed, trying simulation")
    simulated_answer, sim_tokens = simulate_execution(
        current_code, last_error, client, model
    )
    total_tokens += sim_tokens

    if simulated_answer is not None:
        logger.info("Successfully got answer from simulation")
        return str(simulated_answer), total_tokens

    # If we get here, everything failed
    logger.warning("All strategies failed")
    return (
        f"Error: Could not solve problem after all attempts. Last error: {last_error}",
        total_tokens,
    )
