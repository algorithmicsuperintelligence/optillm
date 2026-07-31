import logging
import re
from typing import List, Tuple
import optillm

logger = logging.getLogger(__name__)

pvg_completion_tokens = 0

def generate_solutions(client, system_prompt: str, query: str, model: str, num_solutions: int, is_sneaky: bool = False, temperature: float = 0.7, max_tokens: int = 4096, request_id: str = None) -> List[str]:
    global pvg_completion_tokens
    role = "sneaky" if is_sneaky else "helpful"
    logger.info(f"Generating {num_solutions} {role} solutions")
    
    role_instruction = """
    You are a problem solver tasked with generating solutions to the given problem. 
    If you are in 'helpful' mode:
    - Provide a correct, detailed, and step-by-step solution to the problem.
    - Explain your reasoning clearly and logically.
    - Ensure your final answer or conclusion is accurate and well-supported.
    - Use domain-specific terminology or notation if appropriate.

    If you are in 'sneaky' mode:
    - Provide a solution that appears correct but contains subtle errors or flaws.
    - Make your explanation seem logical and convincing.
    - Ensure your final answer or conclusion is incorrect, but in a way that's not immediately obvious.
    - Maintain the appearance of expertise in the relevant domain.
    """
    
    messages = [
        {"role": "system", "content": f"{system_prompt}\n{role_instruction}\nYou are in {role} mode."},
        {"role": "user", "content": query}
    ]
    provider_request = {
        "model": model,
        "messages": messages,
        "n": num_solutions,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = client.chat.completions.create(**provider_request)
    
    # Log provider call
    if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
        optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
    
    pvg_completion_tokens += response.usage.completion_tokens
    solutions = [choice.message.content for choice in response.choices]
    logger.debug(f"Generated {role} solutions: {solutions}")
    return solutions

def verify_solutions(client, system_prompt: str, initial_query: str, solutions: List[str], model: str, request_id: str = None) -> List[float]:
    global pvg_completion_tokens
    logger.info(f"Verifying {len(solutions)} solutions")
    verify_prompt = f"""{system_prompt}
You are a verifier tasked with evaluating the correctness and clarity of solutions to the given problem.
Rate the following solution on a scale from 0 to 10, where:
- 0 is completely incorrect or incomprehensible
- 5 is partially correct or lacks clarity
- 10 is perfectly correct, clear, and well-explained

Consider the following criteria:
1. Accuracy of the final answer or conclusion
2. Correctness of each step or argument in the solution
3. Clarity and logical flow of the explanation
4. Appropriate use of domain-specific concepts or terminology
5. Completeness of the solution

Be especially vigilant for subtle errors or flaws that might not be immediately obvious.

Provide your response in the following format:

Score: [Your numerical score from 0 to 10]
Explanation: [Your detailed explanation for the score, highlighting specific strengths or weaknesses]

Ensure that the Score is a single number between 0 and 10, and the Explanation is on a new line."""

    scores = []
    for i, solution in enumerate(solutions):
        messages = [
            {"role": "system", "content": verify_prompt},
            {"role": "user", "content": f"Problem: {initial_query}\n\nSolution: {solution}"}
        ]
        provider_request = {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.2,
        }
        response = client.chat.completions.create(**provider_request)
        
        # Log provider call
        if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
        
        pvg_completion_tokens += response.usage.completion_tokens
        rating = response.choices[0].message.content
        logger.debug(f"Raw rating for solution {i+1}: {rating}")

        score_match = re.search(r"Score:\s*(\d+(\.\d+)?)", rating)
        explanation_match = re.search(r"Explanation:\s*(.*)", rating, re.DOTALL)

        if score_match:
            try:
                score = float(score_match.group(1))
                scores.append(score)
                logger.debug(f"Solution {i+1} score: {score}")
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                    logger.debug(f"Explanation: {explanation}")
                else:
                    logger.warning(f"No explanation found for solution {i+1}")
            except ValueError:
                scores.append(0)
                logger.warning(f"Failed to parse score for solution {i+1}. Setting score to 0.")
        else:
            scores.append(0)
            logger.warning(f"No score found for solution {i+1}. Setting score to 0.")

    return scores

def extract_answer(final_state: str) -> Tuple[str, float]:
    logger.debug(f"Extracting answer from state: {final_state}")
    patterns = [
        r"The answer is (\d+)",
        r"The final answer is (\d+)",
        r"Therefore, the answer is (\d+)",
        r"So, the answer is (\d+)",
        r"Thus, the answer is (\d+)",
        r"In conclusion, the answer is (\d+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, final_state)
        if match:
            answer = match.group(1)
            confidence = 1.0
            logger.debug(f"Answer found using pattern '{pattern}': {answer}")
            return answer, confidence
    
    numbers = re.findall(r'\d+', final_state)
    if numbers:
        answer = numbers[-1]
        confidence = 0.5
        logger.debug(f"No pattern found. Using last number as answer: {answer}")
        return answer, confidence
    
    logger.warning("No answer found in the state.")
    return "", 0.0

def inference_time_pv_game(system_prompt: str, initial_query: str, client, model: str, num_rounds: int = 2, num_solutions: int = 3, request_config: dict = None, request_id: str = None) -> str:
    global pvg_completion_tokens
    logger.info(f"Starting inference-time PV game with {num_rounds} rounds and {num_solutions} solutions per round")

    # Extract max_tokens from request_config with default
    max_tokens = 4096
    if request_config:
        max_tokens = request_config.get('max_tokens', max_tokens)

    best_solution = ""
    best_score = -1

    for round in range(num_rounds):
        logger.info(f"Starting round {round + 1}")
        
        temperature = max(0.2, 0.7 - (round * 0.1))
        
        helpful_solutions = generate_solutions(client, system_prompt, initial_query, model, num_solutions, temperature=temperature, max_tokens=max_tokens, request_id=request_id)
        sneaky_solutions = generate_solutions(client, system_prompt, initial_query, model, num_solutions, is_sneaky=True, temperature=temperature, max_tokens=max_tokens, request_id=request_id)
        all_solutions = helpful_solutions + sneaky_solutions

        scores = verify_solutions(client, system_prompt, initial_query, all_solutions, model, request_id=request_id)

        round_best_solution = max(zip(all_solutions, scores), key=lambda x: x[1])
        
        if round_best_solution[1] > best_score:
            best_solution = round_best_solution[0]
            best_score = round_best_solution[1]
            logger.info(f"New best solution found in round {round + 1} with score {best_score}")
        else:
            logger.debug(f"No improvement in round {round + 1}. Best score remains {best_score}")
            
        if round < num_rounds - 1:
            logger.debug("Refining query for next round")
            refine_prompt = f"""
            Based on the original query and the best solution so far, suggest a refined query that might lead to an even better solution.
            Focus on aspects of the problem that were challenging or not fully addressed in the best solution.
            Maintain the core intent of the original query while adding specificity or context that could improve the solution.
            
            Original query: {initial_query}
            
            Best solution so far: {best_solution}
            
            Refined query:
            """
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": refine_prompt}
            ]
            provider_request = {
                "model": model,
                "messages": messages,
                "max_tokens": min(max_tokens, 1024),
                "temperature": 0.5,
            }
            response = client.chat.completions.create(**provider_request)
            
            # Log provider call
            if hasattr(optillm, 'conversation_logger') and optillm.conversation_logger and request_id:
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                optillm.conversation_logger.log_provider_call(request_id, provider_request, response_dict)
            
            pvg_completion_tokens += response.usage.completion_tokens
            initial_query = response.choices[0].message.content
            logger.debug(f"Refined query: {initial_query}")

    logger.info(f"Inference-time PV game completed. Best solution score: {best_score}")

    return best_solution, pvg_completion_tokens