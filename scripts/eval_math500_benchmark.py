import argparse
import json
import os
import logging
import re
from typing import Dict, Optional
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

# Enhanced logging configuration
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('math500_debug.log'),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key="optillm", base_url="http://localhost:8000/v1")
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

SYSTEM_PROMPT = '''You are solving mathematics problems.

Please think step by step.

Important: Always end your solution with the final answer in this format:

\\[
\\boxed{your_answer_here}
\\]

The entire answer should be contained completely within the \\boxed{} command.'''

def load_math500_dataset() -> list[dict]:
    """
    Load the MATH-500 dataset.
    Returns:
        list[dict]: The dataset of problems.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    dataset = dataset["test"]
    logging.debug(f"Dataset size: {len(dataset)}.")
    return dataset

def extract_answer(response: str) -> Optional[str]:
    """Extract the answer from a math solution response."""
    if not response:
        logger.debug("Empty response received")
        return None
    
    # Find the last \boxed{...} in the response
    start_idx = response.rfind('\\boxed{')
    if start_idx == -1:
        logger.debug("No \\boxed{} found in response")
        return None
        
    # Find the matching closing brace
    brace_count = 1
    pos = start_idx + 7  # length of '\boxed{'
    
    while pos < len(response) and brace_count > 0:
        if response[pos] == '{':
            brace_count += 1
        elif response[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count == 0:
        answer = response[start_idx + 7:pos - 1]
        logger.debug(f"Extracted answer: {answer}")
        return answer.strip()
    
    logger.debug("No matching closing brace found")
    return None

def normalize_number(num_str: str) -> str:
    """Helper function to normalize number representation."""
    try:
        # Remove commas, currency symbols, units, and whitespace
        cleaned = re.sub(r'[,\$\\]|\s*(?:cm|m|kg|ft|in|lb|oz|ml|L)$|\s*\\text{[^}]+}', '', num_str).strip()
        
        # Handle leading decimal point
        if cleaned.startswith('.'):
            cleaned = '0' + cleaned
            
        # Convert to float
        num = float(cleaned)
        
        # For small decimals, preserve exact representation
        if abs(num) < 1 and '.' in cleaned:
            # Count original decimal places
            decimal_places = len(cleaned.split('.')[1])
            format_str = f"{{:.{decimal_places}f}}"
            result = format_str.format(num)
        else:
            result = str(num)
        
        logger.debug(f"Normalized number result: {repr(result)}")
        return result
    except Exception as e:
        logger.debug(f"Failed to normalize number: {str(e)}")
        return num_str

def numerically_equal(str1: str, str2: str) -> bool:
    """Compare if two numeric strings represent the same value."""
    try:
        return abs(float(str1) - float(str2)) < 1e-10
    except Exception:
        return False
    
def normalize_fraction(fraction_str: str) -> str:
    """Helper function to normalize fractions."""
    logger.debug(f"Normalizing fraction: {repr(fraction_str)}")
    try:
        # Convert \dfrac to \frac
        fraction_str = fraction_str.replace('\\dfrac', '\\frac')
        
        # Remove all whitespace
        fraction_str = ''.join(fraction_str.split())
        
        # Remove any trailing text
        fraction_str = re.sub(r'\s*\\text{[^}]+}', '', fraction_str)
        
        # Handle mixed brace format first (\frac9{19})
        mixed_brace = re.match(r'^\\frac(\d+)\{(\d+)\}$', fraction_str)
        if mixed_brace:
            num, den = mixed_brace.groups()
            return f"\\frac{{{num}}}{{{den}}}"
        
        # Handle no braces format (\frac12)
        no_braces = re.match(r'^\\frac(\d+)(\d+)$', fraction_str)
        if no_braces:
            num, den = no_braces.groups()
            return f"\\frac{{{num}}}{{{den}}}"
        
        # Handle a/b format
        if '/' in fraction_str and not any(c in fraction_str for c in '\\{}'):
            num, den = fraction_str.split('/')
            return f"\\frac{{{num.strip()}}}{{{den.strip()}}}"
        
        # Handle standard \frac{a}{b}
        standard = re.match(r'^\\frac\{([^{}]+)\}\{([^{}]+)\}$', fraction_str)
        if standard:
            num, den = standard.groups()
            return f"\\frac{{{num}}}{{{den}}}"
            
    except Exception as e:
        logger.debug(f"Failed to normalize fraction: {str(e)}")
        logger.debug(f"Original fraction string: {repr(fraction_str)}")
    return fraction_str

def normalize_matrix_entry(entry: str) -> str:
    """Helper function to normalize a single matrix entry."""
    logger.debug(f"Normalizing matrix entry input: {repr(entry)}")
    
    # Remove all spaces first
    entry = ''.join(entry.split())
    
    # If it's already in simple a/b format, standardize spacing
    if '/' in entry and not any(c in entry for c in '\\{}'):
        if entry.startswith('-'):
            num, den = entry[1:].split('/')
            return f"-{num.strip()}/{den.strip()}"
        else:
            num, den = entry.split('/')
            return f"{num.strip()}/{den.strip()}"
            
    # Convert \dfrac to \frac
    entry = entry.replace('\\dfrac', '\\frac')
    
    # Handle LaTeX fractions
    frac_match = re.match(r'^(-)?\\frac\{(\d+)\}\{(\d+)\}$', entry)
    if frac_match:
        sign, num, den = frac_match.groups()
        sign = sign if sign else ''
        return f"{sign}{num}/{den}"
    
    return entry

def normalize_matrix(matrix_str: str) -> str:
    """Helper function to normalize matrices and vectors."""
    logger.debug(f"Normalizing matrix input: {repr(matrix_str)}")
    try:
        # Remove all whitespace
        matrix_str = ''.join(matrix_str.split())
        
        # Extract the matrix content
        match = re.match(r'^\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}$', matrix_str)
        if not match:
            return matrix_str
            
        content = match.group(1)
        rows = content.split('\\\\')
        
        # Normalize each entry in each row
        normalized_rows = []
        for row in rows:
            if '&' in row:
                entries = [normalize_matrix_entry(entry) for entry in row.split('&')]
            else:
                entries = [normalize_matrix_entry(row)]
            normalized_rows.append('&'.join(entries))
        
        # Reconstruct the matrix
        result = "\\begin{pmatrix}" + "\\\\".join(normalized_rows) + "\\end{pmatrix}"
        logger.debug(f"Normalized matrix result: {repr(result)}")
        return result
        
    except Exception as e:
        logger.debug(f"Failed to normalize matrix: {str(e)}")
        return matrix_str

def normalize_algebraic_expression(expr: str) -> str:
    """Helper function to normalize algebraic expressions."""
    logger.debug(f"Normalizing algebraic expression: {repr(expr)}")
    try:
        # Remove all whitespace
        expr = ''.join(expr.split())
        
        # Handle simple monomial with exponent (e.g., 5r^5)
        monomial_match = re.match(r'^(-?\d*\.?\d*)?([a-zA-Z])(?:\^(-?\d+))?$', expr)
        if monomial_match:
            coeff, var, exp = monomial_match.groups()
            coeff = coeff if coeff and coeff not in ['+', '-'] else ('1' if not coeff else '-1')
            exp = exp if exp else '1'
            if coeff == '1' and exp == '1':
                result = var
            elif coeff == '1':
                result = f"{var}^{exp}"
            elif coeff == '-1' and exp == '1':
                result = f"-{var}"
            elif coeff == '-1':
                result = f"-{var}^{exp}"
            elif exp == '1':
                result = f"{coeff}{var}"
            else:
                result = f"{coeff}{var}^{exp}"
            logger.debug(f"Matched as monomial with exponent: {repr(result)}")
            return result.lower()
            
        # Special case: If it's a single term with π
        pi_term_match = re.match(r'^(-?\d*\.?\d*)\\?pi$', expr)
        if pi_term_match:
            coeff = pi_term_match.group(1)
            if not coeff or coeff == '-':
                coeff = '-1' if coeff == '-' else '1'
            return f"{coeff}\\pi"
            
        # Handle fractions with π
        frac_pi_match = re.match(r'^\\frac{([^{}]+)}{([^{}]+)}\\?pi$', expr)
        if frac_pi_match:
            num, den = frac_pi_match.groups()
            return f"\\frac{{{num}}}{{{den}}}\\pi"
        
        # Handle basic fractions
        frac_match = re.match(r'^\\frac{([^{}]+)}{([^{}]+)}$', expr)
        if frac_match:
            num, den = frac_match.groups()
            return f"\\frac{{{num}}}{{{den}}}"
        
        # Split into terms (handle both + and -)
        terms = []
        current_term = ""
        for i, char in enumerate(expr):
            if char in ['+', '-'] and i > 0:
                if current_term:
                    terms.append(current_term)
                current_term = char
            else:
                current_term += char
        if current_term:
            terms.append(current_term)
        
        # If it's just a single number, return normalized version
        if len(terms) == 1 and re.match(r'^-?[\d,]+$', terms[0]):
            return normalize_number(terms[0])
            
        # Process each term and sort
        processed_terms = []
        for term in terms:
            # Handle leading + if present
            if term.startswith('+'):
                term = term[1:]
                
            # Add implicit + for positive terms
            if not term.startswith('-'):
                term = '+' + term
                
            # Separate coefficient and variable parts
            match = re.match(r'^([+-])?\s*(\d*\.?\d*)?([a-zA-Z](?:\^\d+)?)?$', term)
            if match:
                sign, coeff, var = match.groups()
                # Handle default coefficients
                if not coeff and var:
                    coeff = '1'
                elif not coeff:
                    coeff = '0'
                # Create standardized term
                processed_terms.append((sign, float(coeff), var or ''))
        
        # Sort terms: variables first (in alphabetical order), then constants
        processed_terms.sort(key=lambda x: (not bool(x[2]), x[2], -x[1]))
        
        # Reconstruct the expression
        result = ""
        for sign, coeff, var in processed_terms:
            if coeff == 0:
                continue
            term = ""
            if coeff == 1 and var:
                term = var
            elif coeff == -1 and var:
                term = f"-{var}"
            elif var:
                term = f"{coeff}{var}"
            else:
                term = str(coeff)
            
            if result and term[0] != '-':
                result += '+'
            result += term
        
        logger.debug(f"Normalized algebraic expression result: {repr(result)}")
        return result.lower()
    except Exception as e:
        logger.debug(f"Failed to normalize algebraic expression: {str(e)}")
        return expr.lower()  # Return lowercased original if normalization fails
    
def normalize_interval_bound(bound: str) -> str:
    """Helper function to normalize interval bounds."""
    logger.debug(f"Normalizing interval bound: {repr(bound)}")
    
    # Handle infinity
    if '\\infty' in bound:
        sign = '-' if bound.startswith('-') else ''
        return f"{sign}\\infty"
        
    # For other bounds, use regular answer normalization
    return normalize_answer(bound) or bound

def normalize_interval(interval_str: str) -> str:
    """Helper function to normalize intervals."""
    logger.debug(f"Normalizing interval: {repr(interval_str)}")
    try:
        # Remove all whitespace first
        interval_str = ''.join(interval_str.split())
        
        # Extract the interval content, handling \left and \right
        # Fixed regex to avoid nested set warning by using explicit character classes
        match = re.match(r'^\\left?([\[\(])(.*?),(.*?)\\right?([\]\)])$', interval_str)
        if not match:
            # Try without \left and \right
            match = re.match(r'^([\[\(])(.*?),(.*?)([\]\)])$', interval_str)
            if not match:
                return interval_str
                
        left_bracket, left_bound, right_bound, right_bracket = match.groups()
        
        # Normalize each bound
        norm_left = normalize_interval_bound(left_bound)
        norm_right = normalize_interval_bound(right_bound)
        
        # Reconstruct the interval
        result = f"\\left{left_bracket}{norm_left},{norm_right}\\right{right_bracket}"
        logger.debug(f"Normalized interval result: {repr(result)}")
        return result
        
    except Exception as e:
        logger.debug(f"Failed to normalize interval: {str(e)}")
        return interval_str
    
def normalize_ordered_tuple(tuple_str: str) -> str:
    """Helper function to normalize ordered tuples/lists of numbers."""
    logger.debug(f"Normalizing tuple: {repr(tuple_str)}")
    try:
        # First standardize \dfrac to \frac
        tuple_str = tuple_str.replace('\\dfrac', '\\frac')
        
        # Remove \left and \right
        tuple_str = tuple_str.replace('\\left', '').replace('\\right', '')
        
        # Remove all spaces and backslash spaces
        tuple_str = re.sub(r'\\?\s+', '', tuple_str)
        
        # Remove outer parentheses and split by commas
        inner = tuple_str.strip('()')
        parts = inner.split(',')
        
        # Normalize each part
        normalized_parts = []
        for part in parts:
            norm_part = normalize_answer(part.strip())
            if not norm_part:  # If any part fails to normalize, return None
                logger.debug(f"Failed to normalize part: {part}")
                return None
            normalized_parts.append(norm_part)
            
        # Always reconstruct with standard format (using parentheses)
        result = f"({','.join(normalized_parts)})"
        logger.debug(f"Normalized tuple result: {repr(result)}")
        return result
    except Exception as e:
        logger.debug(f"Failed to normalize tuple: {str(e)}")
        return None

def normalize_answer(answer: str) -> str:
    """Normalize the answer string for comparison."""
    logger.debug(f"Normalizing answer: {repr(answer)}")
    
    if answer is None:
        logger.debug("Received None answer")
        return ""
    
    # Remove \text{} with units first
    answer = re.sub(r'\\text{[^}]+(?:inches|feet|meters|cm|m|kg|ft|in|lb|oz|ml|L|per|second|minute|hour)[^}]*}', '', answer)
    

    # Remove all whitespace first but preserve backslash space temporarily
    answer = re.sub(r'(?<!\\)\s+', '', answer)
    logger.debug(f"After initial whitespace removal: {repr(answer)}")
    
    # Then handle ordered pairs/tuples with potential \left, \right
    ordered_pair_match = re.match(r'^(?:\\left)?\((.*?)(?:\\right)?\)$', answer)
    if ordered_pair_match:
        content = ordered_pair_match.group(1)
        # Split by comma and normalize each part
        parts = content.split(',')
        normalized_parts = []
        for part in parts:
            # Remove any remaining backslash spaces
            part = re.sub(r'\\?\s+', '', part)
            norm_part = normalize_answer(part)
            if norm_part is None:
                return None
            normalized_parts.append(norm_part)
        return f"({','.join(normalized_parts)})"
    
    # Remove all whitespace
    answer = ''.join(answer.split())
    logger.debug(f"After whitespace removal: {repr(answer)}")
    
    if not answer:
        logger.debug("Answer became empty after whitespace removal")
        return None
    
    # Handle plus-minus expressions first
    # This will match both forms: "a \pm b" and "a - b"
    pm_match = re.match(r'^(.*?)(?:\\pm|-)(.*?)$', answer)
    if pm_match:
        left, right = pm_match.groups()
        # Normalize both sides
        norm_left = normalize_answer(left) if left else ""
        norm_right = normalize_answer(right) if right else ""
        if norm_left or norm_right:  # If either side normalized successfully
            # Always use \pm in the normalized form
            result = f"{norm_left}\\pm{norm_right}"
            logger.debug(f"Matched as plus-minus expression: {repr(result)}")
            return result
    
    # Handle trigonometric functions
    trig_match = re.match(r'^\\(?:sin|cos|tan|cot|sec|csc)\s*([a-zA-Z])$', answer)
    if trig_match:
        variable = trig_match.group(1)
        # Get the function name without the backslash
        func_name = re.match(r'^\\(.*?)(?:\s|$)', answer).group(1)
        result = f"\\{func_name}{variable}"
        logger.debug(f"Matched as trigonometric function: {repr(result)}")
        return result

    # Handle text-only answers first (including multiple choice)
    text_match = re.match(r'^(?:\\text{)?([A-Za-z]+)(?:})?$', answer)
    if text_match:
        result = text_match.group(1).lower()
        logger.debug(f"Matched as text answer: {repr(result)}")
        return result

    # Handle intervals first (with or without \left and \right)
    if (answer.startswith('\\left[') or answer.startswith('\\left(') or 
        answer.startswith('[') or answer.startswith('(')) and \
       (answer.endswith('\\right]') or answer.endswith('\\right)') or 
        answer.endswith(']') or answer.endswith(')')):
        result = normalize_interval(answer)
        if result:
            logger.debug(f"Matched as interval: {repr(result)}")
            return result
    
    # Handle matrices/vectors
    if answer.startswith('\\begin{pmatrix}') and answer.endswith('\\end{pmatrix}'):
        result = normalize_matrix(answer)
        if result:
            logger.debug(f"Matched as matrix: {repr(result)}")
            return result
    
    # Normalize all fraction commands to \frac first
    answer = answer.replace('\\dfrac', '\\frac')

    # Handle fractions (both \frac and \dfrac)
    if '\\frac' in answer or '\\dfrac' in answer or '/' in answer:
        result = normalize_fraction(answer)
        if result:
            logger.debug(f"Matched as fraction: {repr(result)}")
            return result

    # Handle negative square roots first (before other square root handling)
    neg_sqrt_match = re.match(r'^-\\sqrt\{?(\d+)\}?$', answer)
    if neg_sqrt_match:
        num = neg_sqrt_match.group(1)
        result = f"-\\sqrt{{{num}}}"
        logger.debug(f"Matched as negative square root: {repr(result)}")
        return result

    # Handle direct square root expressions first
    logger.debug("Checking for square root pattern...")
    sqrt_match = re.match(r'^(\d*)?\\sqrt\{?(\d+)\}?$', answer)
    if sqrt_match:
        coeff, num = sqrt_match.groups()
        coeff = coeff if coeff else '1'
        if coeff == '1':
            result = f"\\sqrt{{{num}}}"
        else:
            result = f"{coeff}\\sqrt{{{num}}}"
        logger.debug(f"Matched as pure square root: {repr(result)}")
        return result

    # Now handle coefficient with square root
    sqrt_with_coeff_match = re.match(r'^(\d+)\\sqrt\{?(\d+)\}?$', answer)
    if sqrt_with_coeff_match:
        coeff, num = sqrt_with_coeff_match.groups()
        result = f"{coeff}\\sqrt{{{num}}}"
        logger.debug(f"Matched as coefficient with square root: {repr(result)}")
        return result
    
    # Handle numbers with base subscripts
    base_match = re.match(r'^(\d+)(?:_\{?(\d+)\}?|_(\d+))$', answer)
    if base_match:
        number, base1, base2 = base_match.groups()
        base = base1 if base1 else base2
        result = f"{number}_{base}"
        logger.debug(f"Matched as base number: {repr(result)}")
        return result

    # Handle numbers with percentage sign first
    percent_match = re.match(r'^(\d+(?:\.\d*)?)\s*\\?%$', answer)
    if percent_match:
        number = percent_match.group(1)
        result = normalize_number(number)
        logger.debug(f"Matched as percentage: {repr(result)}")
        return result
    
    # Handle numbers with units (including LaTeX spaces and comma-separated units)
    unit_match = re.match(r'^(\d+(?:\.\d*)?)\s*(?:(?:\\[,\s])|,)?\s*(?:\\\\)?(?:\\text{(\w+)}|\\?(?:cm|m|kg|ft|in|lb|oz|ml|L))$', answer)
    if unit_match:
        number = unit_match.group(1)
        result = normalize_number(number)
        logger.debug(f"Matched as number with unit: {repr(result)}")
        return result
    
    # Try to handle currency values first
    currency_match = re.match(r'^\\?\$?([\d,]+\.?\d*)$', answer)
    if currency_match:
        result = normalize_number(currency_match.group(1))
        logger.debug(f"Matched as currency: {repr(result)}")
        return result
    
    # Try to handle pure numbers with commas first
    if re.match(r'^-?[\d,]+$', answer):
        result = normalize_number(answer)
        logger.debug(f"Matched as number: {repr(result)}")
        return result
    
    # Try to extract numeric value with optional units
    unit_match = re.match(r'^(-?[\d,]+(?:\.\d*)?)\s*(?:\\(?:mbox|text|hbox|displaystyle)\{[^}]+\})?(?:\^?\d)?$', answer)
    if unit_match:
        result = normalize_number(unit_match.group(1))
        logger.debug(f"Matched as number with units: {repr(result)}")
        return result
    
    # Handle multiple choice answers
    mc_match = re.match(r'^\\text{\(?([A-Za-z])\)?}$|^\(?([A-Za-z])\)?$', answer)
    if mc_match:
        result = (mc_match.group(1) or mc_match.group(2)).lower()
        logger.debug(f"Matched as multiple choice: {repr(result)}")
        return result
    
    # Handle degrees
    degree_match = re.match(r'^(-?[\d,]+(?:\.\d*)?)\s*(?:(?:\^?\\circ)|(?:{\\circ})|(?:°))?$', answer)
    if degree_match:
        result = normalize_number(degree_match.group(1))
        logger.debug(f"Matched as degrees: {repr(result)}")
        return result
    
    # Remove \text{} command without changing content FIRST
    answer = re.sub(r'\\text{([^{}]+)}', r'\1', answer)
    logger.debug(f"After \\text removal: {repr(answer)}")
    
    # Try to handle algebraic expressions
    try:
        result = normalize_algebraic_expression(answer)
        logger.debug(f"Normalized as algebraic expression: {repr(result)}")
        return result
    except Exception:
        logger.debug("Failed to normalize as algebraic expression")
        pass
    
    # Remove \left and \right commands
    answer = answer.replace('\\left', '').replace('\\right', '')
    
    # Remove any remaining extra backslashes before common symbols
    answer = answer.replace('\\left', '').replace('\\right', '')
    answer = answer.replace('\\(', '(').replace('\\)', ')')
    answer = answer.replace('\\[', '[').replace('\\]', ']')
    answer = answer.replace('\\{', '{').replace('\\}', '}')
    
    # Normalize square roots consistently
    answer = re.sub(r'\\sqrt\{?(\d+)\}?', r'\\sqrt{\1}', answer)
    answer = re.sub(r'\\sqrt{([^{}]+)}', r'\\sqrt\1', answer)
    
    # Handle percentage notation
    if re.match(r'^\d+\\%$', answer) or re.match(r'^\d+$', answer):
        answer = re.sub(r'\\%$', '', answer)
    
    # Handle \text{} command again in case it was nested
    answer = re.sub(r'\\text{([^{}]+)}', r'\1', answer)
    
    # Strip unnecessary outer braces
    while len(answer) >= 2 and answer[0] == '{' and answer[-1] == '}':
        if '\\frac' in answer:
            break
        answer = answer[1:-1]
    
    result = answer.lower()
    logger.debug(f"Final normalized result: {repr(result)}")
    return result if result else None

def compare_answers(correct_answer: str, predicted_answer: Optional[str]) -> bool:
    """Compare the correct answer with the predicted answer."""
    logger.debug(f"Comparing answers - Correct: {repr(correct_answer)}, Predicted: {repr(predicted_answer)}")
    
    if predicted_answer is None:
        logger.debug("Predicted answer is None")
        return False
    
    # Try numerical comparison first
    if numerically_equal(correct_answer, predicted_answer):
        return True
        
    normalized_correct = normalize_answer(correct_answer)
    normalized_predicted = normalize_answer(predicted_answer)
    
    logger.debug(f"Normalized answers - Correct: {repr(normalized_correct)}, Predicted: {repr(normalized_predicted)}")
    
    # If either normalization returns None or empty string, answers don't match
    if not normalized_correct or not normalized_predicted:
        logger.debug("One or both normalized answers are None or empty")
        return False
        
    # If both answers became empty strings, they don't match
    if normalized_correct == "" and normalized_predicted == "":
        logger.debug("Both answers normalized to empty strings")
        return False
    
    # For intervals, they must match exactly (including brackets)
    if ('\\left[' in normalized_correct or '\\left(' in normalized_correct) and \
       ('\\left[' in normalized_predicted or '\\left(' in normalized_predicted):
        result = normalized_correct == normalized_predicted
        logger.debug(f"Interval comparison result: {result}")
        return result
    
    result = normalized_correct == normalized_predicted
    logger.debug(f"Comparison result: {result}")
    return result

def get_llm_response(problem: str, model: str) -> str:
    """
    Get response from the LLM for a given problem.
    
    Args:
        problem (str): The problem text
        model (str): The model identifier
        
    Returns:
        str: Model's response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.6,  # Lower temperature for more consistent answers
            messages=[
                {"role": "user", "content": SYSTEM_PROMPT + "\n" + problem}
            ],
            max_tokens=8192, # for thinking models, we need to use a lot more tokens
            # extra_body = {
            #     "decoding" : "thinkdeeper",
            # }
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return ""

def load_existing_results(filename: str) -> list[Dict]:
    """Load existing results from file if it exists."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_result(filename: str, result: Dict):
    """Save a single result to the results file."""
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def analyze_results(results: list[Dict]):
    """
    Analyze and print summary statistics of the results.
    
    Args:
        results (list[Dict]): List of evaluation results
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    print("\n=== Results Summary ===")
    print(f"Total problems: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    print("\n=== Incorrect Problems ===")
    for r in results:
        if not r['is_correct']:
            print(f"Problem {r['index']}:")
            print(f"Expected: {r['correct_answer']}")
            print(f"Predicted: {r['predicted_answer']}")
            print("---")

def main(model: str):
    """Main evaluation function."""
    os.makedirs("results", exist_ok=True)
    results_file = f"evaluation_results_math500_{model.replace('/', '_')}.json"
    
    dataset = load_math500_dataset()
    existing_results = load_existing_results(results_file)
    
    # Create a set of already processed indexes for efficient lookup
    processed_indexes = {result['index'] for result in existing_results}
    
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating problems")):
        # Skip if this index has already been processed
        if idx in processed_indexes:
            continue
            
        problem_text = item['problem']
        correct_answer = item['answer']
        
        # Get model's response
        response = get_llm_response(problem_text, model)
        predicted_answer = extract_answer(response)

        # Compare answers using the new comparison function
        is_correct = compare_answers(correct_answer, predicted_answer)
        
        result = {
            "index": idx,
            "problem": problem_text,
            "response": response,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        }
        save_result(results_file, result)
    
    final_results = load_existing_results(results_file)
    analyze_results(final_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on MATH-500 problems")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)")
    args = parser.parse_args()
    
    main(args.model)
