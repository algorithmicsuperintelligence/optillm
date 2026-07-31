"""
SELF-DISCOVER Framework Implementation

This module implements the SELF-DISCOVER framework for automatically discovering
task-intrinsic reasoning structures.
"""

import json
import logging
import re
from typing import List, Dict, Any
from .reasoning_modules import get_all_modules, get_module_descriptions

logger = logging.getLogger(__name__)

class SelfDiscover:
    """
    Implementation of the SELF-DISCOVER framework.

    The framework operates in two stages:
    1. Stage 1: Discover task-specific reasoning structure (SELECT, ADAPT, IMPLEMENT)
    2. Stage 2: Use discovered structure to solve problem instances
    """

    def __init__(self, client, model: str, max_tokens: int = 16382, request_config: Dict[str, Any] = None):
        self.client = client
        self.model = model
        # Read max_completion_tokens (preferred) or max_tokens (deprecated) from request_config
        if request_config:
            self.max_tokens = request_config.get('max_completion_tokens') or request_config.get('max_tokens', max_tokens)
        else:
            self.max_tokens = max_tokens
        self.reasoning_modules = get_all_modules()
        self.completion_tokens = 0
        
    def discover_reasoning_structure(self, task_description: str, task_examples: List[str] = None) -> Dict[str, Any]:
        """
        Stage 1: Discover reasoning structure for the given task.
        
        Args:
            task_description: Description of the task type
            task_examples: Optional examples of the task (without labels)
            
        Returns:
            Dict containing the discovered reasoning structure
        """
        logger.info("Starting SELF-DISCOVER reasoning structure discovery")
        
        # Step 1: SELECT relevant reasoning modules
        selected_modules = self._select_modules(task_description, task_examples)
        logger.info(f"Selected {len(selected_modules)} reasoning modules")
        
        # Step 2: ADAPT modules to be task-specific
        adapted_modules = self._adapt_modules(selected_modules, task_description, task_examples)
        logger.info("Adapted modules to be task-specific")
        
        # Step 3: IMPLEMENT structured reasoning plan
        reasoning_structure = self._implement_structure(adapted_modules, task_description, task_examples)
        logger.info("Implemented reasoning structure")
        
        return {
            "selected_modules": selected_modules,
            "adapted_modules": adapted_modules,
            "reasoning_structure": reasoning_structure,
            "completion_tokens": self.completion_tokens
        }
    
    def _select_modules(self, task_description: str, task_examples: List[str] = None) -> List[Dict[str, Any]]:
        """SELECT: Choose relevant reasoning modules for the task."""
        
        module_descriptions = get_module_descriptions()
        modules_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(module_descriptions)])
        
        examples_text = ""
        if task_examples:
            examples_text = "\n\nTask examples:\n" + "\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(task_examples)])
        
        select_prompt = f"""You are an expert in problem-solving and reasoning. Given a task description and available reasoning modules, select the most relevant modules that would be useful for solving this type of task.

Task description: {task_description}{examples_text}

Available reasoning modules:
{modules_text}

Instructions:
1. Analyze the task and identify what types of reasoning would be most helpful
2. Select 3-7 reasoning modules that are most relevant for this task
3. Consider both the complexity of the task and the complementary nature of different modules
4. Avoid selecting too many similar modules
5. IMPORTANT: Respond ONLY with a valid JSON array of numbers

Example response format: [1, 5, 9, 15, 23]

Selected modules (JSON array only):"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": select_prompt}],
            max_tokens=1024,
            temperature=0.3
        )

        self.completion_tokens += response.usage.completion_tokens

        try:
            # Check for truncated or empty response
            if (response is None or
                not response.choices or
                response.choices[0].message.content is None or
                response.choices[0].finish_reason == "length"):
                logger.warning("Response truncated or empty in module selection, using fallback modules")
                return self.reasoning_modules[:5]

            # Extract JSON from response
            response_text = response.choices[0].message.content.strip()
            # Look for JSON array in the response
            json_match = re.search(r'\[[\d,\s]+\]', response_text)
            if json_match:
                selected_indices = json.loads(json_match.group(0))
            else:
                # Fallback: extract numbers from response
                numbers = re.findall(r'\b(\d+)\b', response_text)
                selected_indices = [int(n) for n in numbers[:7]]  # Limit to 7 modules
            
            # Convert to module objects (1-indexed to 0-indexed)
            selected_modules = []
            for idx in selected_indices:
                if 1 <= idx <= len(self.reasoning_modules):
                    selected_modules.append(self.reasoning_modules[idx-1])
            
            return selected_modules[:7]  # Ensure we don't exceed reasonable limit
            
        except Exception as e:
            logger.warning(f"Error parsing selected modules: {e}")
            # Fallback to first few modules
            return self.reasoning_modules[:5]
    
    def _adapt_modules(self, selected_modules: List[Dict[str, Any]], task_description: str, task_examples: List[str] = None) -> List[str]:
        """ADAPT: Rephrase modules to be more task-specific."""
        
        modules_text = "\n".join([f"- {module['description']}" for module in selected_modules])
        
        examples_text = ""
        if task_examples:
            examples_text = "\n\nTask examples:\n" + "\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(task_examples)])
        
        adapt_prompt = f"""You are an expert in adapting general reasoning strategies to specific tasks. Given the selected reasoning modules and task description, rephrase each module to be more specific and tailored to this particular type of task.

Task description: {task_description}{examples_text}

Selected reasoning modules:
{modules_text}

Instructions:
1. For each module, rephrase the description to be more specific to this task
2. Keep the core reasoning approach but make it more actionable for this specific type of problem
3. Use terminology and concepts relevant to the task domain
4. Make the adapted descriptions more concrete and specific

Provide the adapted modules as a numbered list:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": adapt_prompt}],
            max_tokens=2048,
            temperature=0.3
        )

        self.completion_tokens += response.usage.completion_tokens

        # Check for truncated or empty response
        if (response is None or
            not response.choices or
            response.choices[0].message.content is None or
            response.choices[0].finish_reason == "length"):
            logger.warning("Response truncated or empty in module adaptation, using generic descriptions")
            # Return generic adapted versions of the selected modules
            return [module.get('description', 'Apply reasoning to solve the problem') for module in selected_modules]

        response_text = response.choices[0].message.content.strip()
        
        # Extract adapted modules from numbered list
        adapted_modules = []
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):
                # Remove the number prefix
                adapted_desc = re.sub(r'^\d+\.\s*', '', line)
                adapted_modules.append(adapted_desc)
        
        return adapted_modules
    
    def _implement_structure(self, adapted_modules: List[str], task_description: str, task_examples: List[str] = None) -> Dict[str, Any]:
        """IMPLEMENT: Create a structured reasoning plan in JSON format."""
        
        modules_text = "\n".join([f"{i+1}. {module}" for i, module in enumerate(adapted_modules)])
        
        examples_text = ""
        if task_examples:
            examples_text = "\n\nTask examples:\n" + "\n".join([f"Example {i+1}: {ex}" for i, ex in enumerate(task_examples)])
        
        # Provide a demonstration of a reasoning structure
        demo_structure = """{
    "problem_analysis": "Analyze the core components and requirements",
    "approach_selection": "Choose the most appropriate solution method",
    "step_by_step_solution": {
        "step_1": "First logical step with clear reasoning",
        "step_2": "Second step building on previous results", 
        "step_3": "Continue logical progression"
    },
    "verification": "Check the solution for accuracy and completeness",
    "final_answer": "Present the final result clearly"
}"""
        
        implement_prompt = f"""You are an expert in creating structured reasoning plans. Given the adapted reasoning modules for a specific task, create a detailed JSON reasoning structure that can be followed step-by-step to solve instances of this task.

Task description: {task_description}{examples_text}

Adapted reasoning modules:
{modules_text}

Example of a reasoning structure format:
{demo_structure}

Instructions:
1. Create a JSON structure that operationalizes the adapted reasoning modules
2. The structure should be specific enough to guide step-by-step reasoning
3. Include clear field names that indicate what should be filled in each step
4. Make it actionable - each field should represent a concrete reasoning step
5. Ensure the structure flows logically from problem understanding to final answer
6. The structure should be comprehensive enough to handle the complexity of the task

7. IMPORTANT: Return ONLY valid JSON with double quotes around all property names and string values
8. Do not include any text before or after the JSON structure

Valid JSON reasoning structure:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": implement_prompt}],
            max_tokens=2048,
            temperature=0.3
        )

        self.completion_tokens += response.usage.completion_tokens

        # Check for truncated or empty response
        if (response is None or
            not response.choices or
            response.choices[0].message.content is None or
            response.choices[0].finish_reason == "length"):
            logger.warning("Response truncated or empty in structure implementation, using fallback structure")
            # Return the fallback structure directly
            return {
                "problem_understanding": "Analyze and understand the problem requirements",
                "solution_approach": "Determine the best approach based on problem characteristics",
                "step_by_step_reasoning": "Work through the problem systematically",
                "verification": "Verify the solution is correct and complete",
                "final_answer": "State the final answer clearly"
            }

        response_text = response.choices[0].message.content.strip()
        
        # Extract and parse JSON from response with improved error handling
        return self._parse_json_structure(response_text)
    
    def _parse_json_structure(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON structure with robust error handling and cleanup."""
        
        # Define fallback structure
        fallback_structure = {
            "problem_understanding": "Analyze and understand the problem requirements",
            "solution_approach": "Determine the best approach based on problem characteristics", 
            "step_by_step_reasoning": "Work through the problem systematically",
            "verification": "Verify the solution is correct and complete",
            "final_answer": "State the final answer clearly"
        }
        
        # Try multiple JSON extraction and parsing strategies
        strategies = [
            self._extract_json_strategy_1,
            self._extract_json_strategy_2,
            self._extract_json_strategy_3,
            self._clean_and_parse_strategy
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                structure = strategy(response_text)
                if structure and isinstance(structure, dict) and len(structure) > 0:
                    logger.debug(f"Successfully parsed JSON using strategy {i}")
                    return structure
            except Exception as e:
                logger.debug(f"Strategy {i} failed: {e}")
                continue
        
        logger.warning("All JSON parsing strategies failed. Using fallback structure.")
        logger.debug(f"Raw response that failed to parse: {response_text[:500]}...")
        return fallback_structure
    
    def _extract_json_strategy_1(self, text: str) -> Dict[str, Any]:
        """Strategy 1: Find first complete JSON object with balanced braces."""
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No opening brace found")
        
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            raise ValueError("Unbalanced braces")
        
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    
    def _extract_json_strategy_2(self, text: str) -> Dict[str, Any]:
        """Strategy 2: Use regex with non-greedy matching."""
        # Look for JSON object with non-greedy matching
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if not json_match:
            raise ValueError("No JSON object found with regex")
        
        json_str = json_match.group(0)
        return json.loads(json_str)
    
    def _extract_json_strategy_3(self, text: str) -> Dict[str, Any]:
        """Strategy 3: Extract between ```json``` code blocks."""
        patterns = [
            r'```json\s*([^`]+)```',
            r'```\s*([^`]+)```',
            r'`([^`]+)`'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                try:
                    return json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    continue
        
        raise ValueError("No valid JSON found in code blocks")
    
    def _clean_and_parse_strategy(self, text: str) -> Dict[str, Any]:
        """Strategy 4: Clean common formatting issues and parse."""
        # Find JSON-like content
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON-like content found")
        
        json_str = json_match.group(0)
        
        # Common cleanup operations
        cleanups = [
            # Fix single quotes to double quotes (but be careful about apostrophes)
            (r"(?<!\\)'([^']*)'(?=\s*[,}])", r'"\1"'),
            # Fix unquoted property names
            (r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
            # Fix trailing commas
            (r',\s*([}\]])', r'\1'),
            # Fix extra commas
            (r',,+', r','),
        ]
        
        for pattern, replacement in cleanups:
            json_str = re.sub(pattern, replacement, json_str)
        
        # Try parsing the cleaned JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # One more attempt: try to fix the specific error location
            if "line 1 column 2" in str(e):
                # Common issue: extra characters at start
                json_str = re.sub(r'^[^{]*', '', json_str)
                return json.loads(json_str)
            else:
                raise e
    
    def solve_with_structure(self, problem: str, reasoning_structure: Dict[str, Any]) -> str:
        """
        Stage 2: Use the discovered reasoning structure to solve a specific problem.
        """
        
        structure_text = json.dumps(reasoning_structure, indent=2)
        
        solve_prompt = f"""Follow the step-by-step reasoning structure below to solve the given problem. Fill in each field with your reasoning and analysis, then provide your final answer.

Reasoning Structure:
{structure_text}

Problem to solve: {problem}

Instructions:
1. Work through each field in the reasoning structure systematically
2. Provide detailed reasoning for each step
3. Use the structure to guide your thinking process
4. Ensure your reasoning is logical and well-supported
5. Wrap your internal reasoning in <think> tags
6. Provide a clear final answer after your reasoning

<think>
[Follow the reasoning structure step by step here]
</think>

Based on my systematic analysis using the reasoning structure, the answer is:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": solve_prompt}],
            max_tokens=self.max_tokens,
            temperature=0.7
        )

        self.completion_tokens += response.usage.completion_tokens

        # Check for truncated or empty response
        if (response is None or
            not response.choices or
            response.choices[0].message.content is None or
            response.choices[0].finish_reason == "length"):
            logger.error("Response truncated or empty when solving with structure. Consider increasing max_tokens.")
            return "Error: Response was truncated due to token limit. Please increase max_tokens or max_completion_tokens."

        return response.choices[0].message.content.strip()
