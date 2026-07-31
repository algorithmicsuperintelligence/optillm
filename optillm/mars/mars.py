"""
MARS: Multi-Agent Reasoning System main orchestration with parallel execution
"""

import asyncio
import logging
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from collections import Counter
from optillm import conversation_logger
from optillm.utils.answer_extraction import extract_answer

from .workspace import MARSWorkspace
from .agent import MARSAgent
from .verifier import MARSVerifier
from .aggregator import MARSAggregator
from .strategy_network import StrategyNetwork
from .prompts import SYNTHESIS_PROMPT
from .answer_extraction import (
    extract_clean_answer,
    wrap_with_thinking_tags,
)

logger = logging.getLogger(__name__)

# Default MARS configuration - simplified with OpenRouter effort parameter
DEFAULT_CONFIG = {
    'num_agents': 3,
    'max_iterations': 5,  # Balanced for quality vs efficiency
    'verification_passes_required': 2,  # Balanced for 5-iteration efficiency
    'consensus_threshold': 2,  # Keep at 2 for 3-agent setup
    'min_verified_solutions': 1,  # Keep minimal requirement
    'max_tokens': 64000,  # Increased default token budget for complex reasoning
    'max_verification_attempts': 3,
    'early_termination': True,
    'use_reasoning_api': True,
    # RSA-inspired aggregation parameters
    'enable_aggregation': True,  # Enable recursive self-aggregation
    'population_size': 6,  # N parameter: maintain larger population for diversity
    'aggregation_size': 3,  # K parameter: number of solutions to aggregate
    'aggregation_loops': 3,  # T parameter: number of aggregation iterations
    # Strategy Network parameters for cross-agent insight sharing
    'enable_strategy_network': True,  # Enable cross-agent strategy sharing
    'strategy_extraction_enabled': True,  # Extract reasoning strategies from solutions
    'cross_agent_enhancement': True,  # Generate enhanced solutions using peer strategies
    # Thinking tags for clean answer extraction
    'use_thinking_tags': True,  # Wrap reasoning in <think></think> tags
    'answer_extraction_mode': 'auto',  # 'auto', 'code', 'math', or 'none'
}

# Lightweight MARS configuration for coding benchmarks (faster, simpler)
LIGHTWEIGHT_CONFIG = {
    'num_agents': 2,  # Reduced from 3
    'max_iterations': 2,  # Reduced from 5 for speed
    'verification_passes_required': 1,  # Reduced from 2
    'consensus_threshold': 1,  # Adjusted for 2-agent setup
    'min_verified_solutions': 1,
    'max_tokens': 4000,  # Much smaller for coding
    'max_verification_attempts': 2,  # Reduced from 3
    'early_termination': True,
    'use_reasoning_api': True,
    # Disable expensive features for coding
    'enable_aggregation': False,  # Skip RSA aggregation
    'enable_strategy_network': False,  # Skip strategy network
    'strategy_extraction_enabled': False,
    'cross_agent_enhancement': False,
    # Thinking tags for clean answer extraction
    'use_thinking_tags': True,  # Wrap reasoning in <think></think> tags
    'answer_extraction_mode': 'auto',  # 'auto', 'code', 'math', or 'none'
}

def multi_agent_reasoning_system(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: dict = None,
    request_id: str = None
) -> Tuple[str, int]:
    """
    Main MARS function implementing multi-agent reasoning with parallel execution

    Args:
        system_prompt: System-level instructions
        initial_query: The problem or task to solve
        client: OpenAI-compatible client for API calls
        model: Model identifier (should support OpenRouter reasoning API)
        request_id: Optional request ID for conversation logging

    Returns:
        Tuple of (final_solution, total_reasoning_tokens)
    """
    return asyncio.run(_run_mars_parallel(
        system_prompt, initial_query, client, model, request_config, request_id
    ))

async def _run_mars_parallel(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: dict = None,
    request_id: str = None
) -> Tuple[str, int]:
    """Async implementation of MARS with parallel execution"""
    start_time = time.time()

    logger.info(f"🚀 MARS INITIALIZATION - Starting MARS with model: {model}")
    logger.info(f"📝 PROBLEM: {initial_query[:200]}{'...' if len(initial_query) > 200 else ''}")

    # Initialize configuration - use lightweight config for coding if max_tokens <= 4000
    use_lightweight = request_config and request_config.get('max_tokens', 64000) <= 4000
    config = LIGHTWEIGHT_CONFIG.copy() if use_lightweight else DEFAULT_CONFIG.copy()

    if use_lightweight:
        logger.info("⚡ CONFIG: Using LIGHTWEIGHT MARS config for coding (fast mode)")

    # Override with mars_config if provided
    if request_config and 'mars_config' in request_config:
        mars_config = request_config['mars_config']
        config.update(mars_config)
        logger.info(f"⚙️  CONFIG: Applied mars_config overrides: {list(mars_config.keys())}")

    # Override max_tokens from request_config if provided (backward compatibility)
    if request_config and 'max_tokens' in request_config:
        config['max_tokens'] = request_config['max_tokens']
        logger.info(f"⚙️  CONFIG: Using max_tokens from request: {config['max_tokens']}")
    else:
        logger.info(f"⚙️  CONFIG: Using default max_tokens: {config['max_tokens']}")

    # Log complete configuration
    logger.info("⚙️  CONFIG: Full MARS configuration:")
    for key, value in config.items():
        logger.info(f"⚙️  CONFIG:   {key}: {value}")

    total_reasoning_tokens = 0

    # Calculate optimal worker count for parallel execution
    max_workers = max(
        config['num_agents'],  # For generation phase
        config['num_agents'] * min(2, config['verification_passes_required'])  # For verification
    )
    logger.info(f"Using {max_workers} parallel workers")

    # Initialize workspace for collaboration
    workspace = MARSWorkspace(initial_query, config)

    # Initialize timing tracking
    phase_times = {}

    try:
        # Phase 1: Initialize Agents
        agents = []
        temperatures = []
        for i in range(config['num_agents']):
            agent = MARSAgent(i, client, model, config)
            agents.append(agent)
            temperatures.append(agent.temperature)

        logger.info(f"🤖 AGENTS: Initialized {len(agents)} agents:")
        for i, temp in enumerate(temperatures):
            effort = agents[i]._get_reasoning_effort()
            logger.info(f"🤖 AGENTS:   Agent {i}: temp={temp}, effort={effort}, max_tokens={config['max_tokens']}")

        # Create thread pool executor for parallel API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Phase 2: Multi-Agent Exploration (parallel)
            phase_start = time.time()
            logger.info(f"📊 PHASE 1: Multi-Agent Exploration - Starting parallel generation with {config['num_agents']} agents")
            exploration_tokens = await _run_exploration_phase_parallel(
                agents, workspace, request_id, executor
            )
            total_reasoning_tokens += exploration_tokens
            phase_times['exploration'] = time.time() - phase_start
            logger.info(f"📊 PHASE 1: Completed in {phase_times['exploration']:.2f}s - Generated {len(workspace.solutions)} solutions, {exploration_tokens} reasoning tokens")

            # Phase 2a: RSA-inspired Aggregation (if enabled)
            if config.get('enable_aggregation', True):
                phase_start = time.time()
                logger.info("📊 PHASE 2a: RSA-inspired Solution Aggregation")
                aggregator = MARSAggregator(client, model, config)
                aggregation_tokens, aggregation_summary = await aggregator.run_aggregation_loops(
                    workspace, request_id, executor
                )
                total_reasoning_tokens += aggregation_tokens
                phase_times['aggregation'] = time.time() - phase_start
                logger.info(f"📊 PHASE 2a: Completed in {phase_times['aggregation']:.2f}s - {aggregation_summary}, {aggregation_tokens} reasoning tokens")

            # Phase 2b: Cross-Agent Strategy Sharing (if enabled)
            if config.get('enable_strategy_network', True):
                phase_start = time.time()
                logger.info("📊 PHASE 2b: Cross-Agent Strategy Network")
                strategy_network = StrategyNetwork(client, model, config)

                # Extract reasoning strategies from agent solutions
                if config.get('strategy_extraction_enabled', True):
                    extracted_strategies = await strategy_network.extract_strategies_from_solutions(
                        workspace, request_id, executor
                    )

                    # Share strategies across agents and generate enhanced solutions
                    if config.get('cross_agent_enhancement', True) and extracted_strategies:
                        await strategy_network.share_strategies_across_agents(
                            workspace, extracted_strategies, request_id, executor
                        )

                        strategy_insights = strategy_network.get_strategy_insights_summary()
                        phase_times['strategy_network'] = time.time() - phase_start
                        logger.info(f"📊 PHASE 2b: Completed in {phase_times['strategy_network']:.2f}s - {strategy_insights}")

            # Phase 3: Verification System (parallel)
            phase_start = time.time()
            logger.info(f"📊 PHASE 3: Verification System - Verifying {len(workspace.solutions)} solutions")
            verifier = MARSVerifier(agents, workspace, config)
            verification_summary = await verifier.verify_solutions_parallel(request_id, executor)
            phase_times['verification'] = time.time() - phase_start
            logger.info(f"📊 PHASE 3: Completed in {phase_times['verification']:.2f}s - {verification_summary}")

            # Phase 4: Iterative Improvement (if needed)
            iteration_count = 0
            improvement_start = time.time()
            while workspace.should_continue_iteration() and iteration_count < config['max_iterations']:
                iteration_count += 1
                iter_start = time.time()
                logger.info(f"📊 PHASE 4: Iterative Improvement - Iteration {iteration_count}/{config['max_iterations']}")

                # Improve unverified solutions (parallel)
                improvement_summary = await verifier.iterative_improvement_parallel(request_id, executor)
                total_reasoning_tokens += improvement_summary['total_reasoning_tokens']

                # Re-verify improved solutions (parallel)
                verification_summary = await verifier.verify_solutions_parallel(request_id, executor)

                iter_time = time.time() - iter_start
                logger.info(f"📊 PHASE 4: Iteration {iteration_count} completed in {iter_time:.2f}s - {improvement_summary}")

                # Check for early termination
                if config['early_termination'] and workspace.has_consensus():
                    logger.info(f"🎯 EARLY TERMINATION: Consensus reached after {iteration_count} iterations")
                    break

                workspace.iteration_count = iteration_count

            if iteration_count > 0:
                phase_times['improvement'] = time.time() - improvement_start
                logger.info(f"📊 PHASE 4: Total improvement time: {phase_times['improvement']:.2f}s")

        # Phase 5: Final Synthesis (sequential - needs all results)
        phase_start = time.time()
        logger.info(f"📊 PHASE 5: Final Synthesis - Processing {len(workspace.solutions)} solutions")

        # Log solution overview before synthesis
        _log_solution_overview(workspace)

        final_solution, synthesis_tokens = _synthesize_final_solution(
            workspace, client, model, config, request_id
        )
        total_reasoning_tokens += synthesis_tokens
        phase_times['synthesis'] = time.time() - phase_start
        logger.info(f"📊 PHASE 5: Completed in {phase_times['synthesis']:.2f}s - Generated {len(final_solution)} char solution, {synthesis_tokens} reasoning tokens")

        # Set final solution in workspace
        workspace.set_final_solution(final_solution)

        # Log comprehensive summary
        total_time = time.time() - start_time
        summary = workspace.get_summary()

        logger.info("🏁 MARS COMPLETION SUMMARY:")
        logger.info(f"🏁   Total execution time: {total_time:.2f}s")
        logger.info(f"🏁   Solutions: {summary['verified_solutions']}/{summary['total_solutions']} verified")
        logger.info(f"🏁   Total reasoning tokens: {total_reasoning_tokens}")
        logger.info(f"🏁   Final solution length: {len(final_solution)} characters")

        # Log phase timing breakdown
        logger.info("🏁 TIMING BREAKDOWN:")
        for phase, duration in phase_times.items():
            percentage = (duration / total_time) * 100
            logger.info(f"🏁   {phase}: {duration:.2f}s ({percentage:.1f}%)")

        # Apply thinking tags if enabled
        if config.get('use_thinking_tags', True):
            try:
                logger.info(f"📝 ANSWER EXTRACTION: Extracting clean answer with mode '{config.get('answer_extraction_mode', 'auto')}'")

                # Extract clean answer from synthesis output
                clean_answer = extract_clean_answer(
                    final_solution,
                    mode=config.get('answer_extraction_mode', 'auto')
                )

                logger.info(f"📝 ANSWER EXTRACTION: Extracted {len(clean_answer)} char answer from {len(final_solution)} char synthesis")

                # Wrap reasoning in thinking tags
                formatted_output = wrap_with_thinking_tags(final_solution, clean_answer)

                logger.info(f"📝 ANSWER EXTRACTION: Final output: {len(formatted_output)} chars (with thinking tags)")
                return formatted_output, total_reasoning_tokens
            except Exception as extract_error:
                # If answer extraction fails, fall back to raw synthesis
                logger.warning(f"⚠️  ANSWER EXTRACTION FAILED: {str(extract_error)}")
                logger.warning(f"⚠️  Falling back to raw synthesis output ({len(final_solution)} chars)")
                return final_solution, total_reasoning_tokens
        else:
            logger.info("📝 ANSWER EXTRACTION: Thinking tags disabled, returning raw synthesis")
            return final_solution, total_reasoning_tokens

    except Exception as e:
        logger.error(f"MARS execution failed: {str(e)}")
        # Return error response
        error_response = f"MARS system encountered an error: {str(e)}\n\nAttempting direct solution approach..."

        # Fallback to single agent approach
        try:
            fallback_agent = MARSAgent(0, client, model, config)
            fallback_solution, fallback_tokens = fallback_agent.generate_solution(initial_query, request_id)
            return fallback_solution.solution, fallback_tokens
        except Exception:
            return error_response, 0

async def _run_exploration_phase_parallel(
    agents: List[MARSAgent],
    workspace: MARSWorkspace,
    request_id: str = None,
    executor: ThreadPoolExecutor = None
) -> int:
    """Run the multi-agent exploration phase with parallel execution"""

    async def generate_solution_async(agent: MARSAgent):
        """Async wrapper for agent solution generation"""
        loop = asyncio.get_event_loop()
        try:
            solution, tokens = await loop.run_in_executor(
                executor,
                agent.generate_solution,
                workspace.problem,
                request_id
            )
            return agent.agent_id, solution, tokens, None
        except Exception as e:
            logger.error(f"Agent {agent.agent_id} failed during exploration: {str(e)}")
            return agent.agent_id, None, 0, e

    # Run all agents in parallel
    tasks = [generate_solution_async(agent) for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_tokens = 0
    successful_solutions = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Agent task failed: {str(result)}")
            continue

        agent_id, solution, tokens, error = result
        if error is None and solution is not None:
            workspace.add_solution(solution)
            total_tokens += tokens
            successful_solutions += 1

            # ENHANCED LOGGING: Log individual agent solution details
            logger.info(f"Agent {agent_id} exploration complete:")
            logger.info(f"  - Solution length: {solution.solution_length} chars")
            logger.info(f"  - Total tokens: {solution.total_tokens}")
            logger.info(f"  - Reasoning tokens: {solution.reasoning_tokens}")
            logger.info(f"  - Confidence: {solution.confidence:.2f}")
            logger.info(f"  - Solution preview: {solution.solution[:200]}...")
        else:
            logger.error(f"Agent {agent_id} generated no solution")

    logger.info(f"Exploration phase complete: {successful_solutions} solutions generated in parallel")
    return total_tokens

def _synthesize_final_solution(
    workspace: MARSWorkspace,
    client,
    model: str,
    config: Dict[str, Any],
    request_id: str = None
) -> Tuple[str, int]:
    """Synthesize the final solution from all agent outputs and verifications"""

    # Get the best verified solution
    best_solution = workspace.get_best_solution()

    if best_solution and best_solution.is_verified:
        logger.info(f"Using verified solution from agent {best_solution.agent_id}")
        return best_solution.solution, 0

    # If no verified solution, try numerical voting first
    logger.info(f"🗳️  VOTING: No verified solutions found, attempting numerical voting on {len(workspace.solutions)} solutions")

    # Enhanced answer extraction using unified math-verify extraction
    numerical_answers = []
    extracted_answers_info = []  # Track all extracted answers for synthesis
    logger.info(f"🗳️  VOTING: Starting unified answer extraction from {len(workspace.solutions)} solutions")

    for i, solution in enumerate(workspace.solutions):
        # Use unified answer extraction with problem context
        extracted_answer = extract_answer(
            solution.solution,
            problem_type="imo",  # Assume IMO context for now
            problem_id=None  # Could be enhanced to detect problem ID
        )

        if extracted_answer is not None:
            logger.info(f"🗳️  VOTING: Agent {solution.agent_id} extracted answer '{extracted_answer}' via unified extraction (confidence: {solution.confidence:.2f})")

            # Math-verify returns a list of all possible matches
            # Iterate through list to find first valid answer
            answers_to_process = []
            if isinstance(extracted_answer, list):
                answers_to_process = extracted_answer
            else:
                answers_to_process = [extracted_answer]

            # Process each answer in the list
            for ans in answers_to_process:
                # Handle both numeric and non-numeric answers
                if isinstance(ans, (int, float)):
                    # Numeric answer - add to numerical voting
                    numerical_answers.append((int(ans), solution))
                    extracted_answers_info.append((str(int(ans)), solution, "unified_numeric"))
                    break  # Use first numeric answer found
                elif isinstance(ans, str) and ans.strip():
                    # Non-numeric answer (formulas, sets, etc.) - store for synthesis
                    extracted_answers_info.append((ans, solution, "unified_formula"))
                    logger.info(f"🗳️  VOTING: Non-numeric answer stored for synthesis: '{ans}'")
                    break  # Use first valid string
                elif isinstance(ans, set):
                    # Set answers (e.g., for Problem 1) - convert to string for synthesis
                    set_str = "{" + ", ".join(map(str, sorted(ans))) + "}"
                    extracted_answers_info.append((set_str, solution, "unified_set"))
                    logger.info(f"🗳️  VOTING: Set answer stored for synthesis: '{set_str}'")
                    break  # Use first set found

            # If no valid answer found after iterating list, log as other type
            if not any(isinstance(ans, (int, float, str, set)) for ans in answers_to_process if isinstance(ans, str) and ans.strip()):
                extracted_answers_info.append((str(extracted_answer), solution, "unified_other"))
                logger.info(f"🗳️  VOTING: Other answer type stored for synthesis: '{extracted_answer}'")
        else:
            logger.info(f"🗳️  VOTING: Agent {solution.agent_id} - no answer extracted via unified extraction (confidence: {solution.confidence:.2f})")

    # Store extracted answers for synthesis use
    workspace._extracted_answers_info = getattr(workspace, '_extracted_answers_info', []) + extracted_answers_info

    # Check for majority vote
    logger.info(f"🗳️  VOTING: Extracted {len(numerical_answers)} numerical answers from {len(workspace.solutions)} solutions")

    if len(numerical_answers) >= 2:
        answer_counts = Counter([ans for ans, _ in numerical_answers])
        most_common_answers = answer_counts.most_common()

        logger.info("🗳️  VOTING: Answer distribution:")
        for answer, count in most_common_answers:
            percentage = (count / len(numerical_answers)) * 100
            agents_with_answer = [sol.agent_id for ans, sol in numerical_answers if ans == answer]
            logger.info(f"🗳️  VOTING:   Answer {answer}: {count}/{len(numerical_answers)} votes ({percentage:.1f}%) - Agents: {agents_with_answer}")

        answer, count = most_common_answers[0]

        # If 2+ agents agree on the same number, use that
        if count >= 2:
            # Find the solution with highest confidence among those with the winning answer
            matching_solutions = [sol for ans, sol in numerical_answers if ans == answer]
            best_solution = max(matching_solutions, key=lambda s: s.confidence)

            logger.info(f"🎆 VOTING SUCCESS: Using majority vote answer {answer} ({count}/{len(numerical_answers)} agents agreed)")
            logger.info(f"🎆 VOTING SUCCESS: Selected solution from agent {best_solution.agent_id} with confidence {best_solution.confidence:.2f}")
            logger.info(f"🎆 VOTING SUCCESS: Solution length: {len(best_solution.solution)} chars")

            # Return the solution with the winning answer (no reasoning tokens since no new API call)
            return best_solution.solution, 0
        else:
            logger.info(f"🗳️  VOTING: No consensus - best answer {answer} only has {count} vote(s), need 2+")
    else:
        logger.info(f"🗳️  VOTING: Insufficient numerical answers for voting ({len(numerical_answers)} < 2)")

    # If no consensus, fall back to synthesis with answer preservation
    logger.info("🤔 VOTING FALLBACK: No numerical consensus found, falling back to answer-preserving synthesis")

    # Log extracted answers for synthesis guidance
    all_extracted = getattr(workspace, '_extracted_answers_info', [])
    if all_extracted:
        logger.info(f"🔍 EXTRACTED ANSWERS SUMMARY: Found {len(all_extracted)} extracted answers:")
        for answer, solution, method in all_extracted:
            logger.info(f"🔍 EXTRACTED ANSWERS SUMMARY:   '{answer}' from Agent {solution.agent_id} via {method}")
    else:
        logger.info("🔍 EXTRACTED ANSWERS SUMMARY: No extracted answers found")

    synthesis_data = workspace.get_synthesis_input()

    # Log synthesis input details
    input_chars = sum(len(sol_data['solution']) for sol_data in synthesis_data['solutions'])
    logger.info(f"🤝 SYNTHESIS INPUT: Processing {len(synthesis_data['solutions'])} solutions")
    logger.info(f"🤝 SYNTHESIS INPUT: Total input characters: {input_chars:,}")
    logger.info(f"🤝 SYNTHESIS INPUT: Verification summary: {synthesis_data['verification_summary']}")

    # Prepare synthesis prompt
    agent_solutions_text = ""
    solutions_used = synthesis_data['solutions'][:3]  # Limit to top 3
    logger.info(f"🤝 SYNTHESIS INPUT: Using top {len(solutions_used)} solutions for synthesis:")

    for i, sol_data in enumerate(solutions_used):
        logger.info(f"🤝 SYNTHESIS INPUT:   Solution {i+1}: Agent {sol_data['agent_id']}, {len(sol_data['solution']):,} chars, confidence {sol_data['confidence']:.2f}")
        agent_solutions_text += f"\nAgent {sol_data['agent_id']} (confidence: {sol_data['confidence']:.2f}):\n"
        agent_solutions_text += sol_data['solution']
        agent_solutions_text += "\n" + "="*50 + "\n"

    synthesis_input_chars = len(agent_solutions_text)
    verification_text = f"Verification Summary: {synthesis_data['verification_summary']}"

    logger.info(f"🤝 SYNTHESIS INPUT: Final synthesis prompt: {synthesis_input_chars:,} characters")

    # Enhanced synthesis prompt with extracted answers
    extracted_answers_text = ""
    all_extracted = getattr(workspace, '_extracted_answers_info', [])
    if all_extracted:
        extracted_answers_text = "\n\nEXTRACTED ANSWERS FROM AGENTS:\n"
        for answer, solution, method in all_extracted:
            extracted_answers_text += f"- Agent {solution.agent_id}: '{answer}' (via {method})\n"
        extracted_answers_text += "\nIMPORTANT: If multiple agents extracted the same answer, prioritize it in your synthesis.\n"
        extracted_answers_text += "Ensure the final answer is clearly formatted and matches the expected answer format.\n"

    synthesis_prompt = SYNTHESIS_PROMPT.format(
        problem=workspace.problem,
        agent_solutions=agent_solutions_text,
        verification_results=verification_text
    ) + extracted_answers_text

    try:
        # Use simplified synthesis with effort parameter
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a mathematical synthesis expert."},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=config['max_tokens'],
            temperature=0.3,  # Lower temperature for synthesis
            timeout=300,
            extra_body={
                "reasoning": {
                    "effort": "high"  # High effort for final synthesis
                }
            }
        )

        # Log provider call if conversation logging is enabled
        if request_id:
            provider_request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a mathematical synthesis expert."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                "max_tokens": config['max_tokens'],
                "temperature": 0.3,
                "extra_body": {
                    "reasoning": {
                        "effort": "high"
                    }
                }
            }
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            conversation_logger.log_provider_call(request_id, provider_request, response_dict)

        final_solution = response.choices[0].message.content.strip()

        # Calculate synthesis compression ratio
        output_chars = len(final_solution)
        compression_ratio = (output_chars / synthesis_input_chars) * 100 if synthesis_input_chars > 0 else 0
        logger.info(f"🤝 SYNTHESIS PROCESSING: Input: {synthesis_input_chars:,} chars → Output: {output_chars:,} chars ({compression_ratio:.1f}% retention)")

        # Extract reasoning tokens from correct nested structure (matching agent.py fix)
        reasoning_tokens = 0
        total_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            total_tokens = getattr(response.usage, 'total_tokens', 0)
            # Check completion_tokens_details first (OpenRouter structure)
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
            # Fallback to direct usage field (standard OpenAI structure)
            if reasoning_tokens == 0:
                reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)

        # ENHANCED LOGGING: Log synthesis details
        logger.info("🤝 SYNTHESIS SUCCESS: Synthesis completed")
        logger.info(f"🤝 SYNTHESIS SUCCESS:   Output solution length: {len(final_solution)} characters")
        logger.info(f"🤝 SYNTHESIS SUCCESS:   Reasoning tokens: {reasoning_tokens}")
        logger.info(f"🤝 SYNTHESIS SUCCESS:   Total tokens: {total_tokens}")
        logger.info(f"🤝 SYNTHESIS SUCCESS:   Solution preview: {final_solution[:200]}...")
        return final_solution, reasoning_tokens

    except Exception as e:
        logger.error(f"🚨 SYNTHESIS ERROR: Synthesis failed: {str(e)}")

        # Fallback: return the solution with highest verification score
        if workspace.solutions:
            fallback_solution = max(workspace.solutions, key=lambda s: s.verification_score)
            logger.info(f"🚑 SYNTHESIS FALLBACK: Using fallback solution from agent {fallback_solution.agent_id}")
            logger.info(f"🚑 SYNTHESIS FALLBACK: Solution length: {len(fallback_solution.solution):,} chars, score: {fallback_solution.verification_score:.2f}")
            return fallback_solution.solution, 0

        logger.error("🚨 SYNTHESIS ERROR: No solutions available for fallback")
        return "Unable to generate solution due to synthesis failure.", 0

def _log_solution_overview(workspace: MARSWorkspace):
    """Log comprehensive overview of all solutions before synthesis"""
    logger.info(f"📋 SOLUTION OVERVIEW: Analyzing {len(workspace.solutions)} solutions before synthesis")

    # Overall statistics
    total_chars = sum(len(sol.solution) for sol in workspace.solutions)
    avg_chars = total_chars / len(workspace.solutions) if workspace.solutions else 0
    verified_solutions = workspace.get_verified_solutions()

    logger.info("📋 SOLUTION OVERVIEW: Statistics:")
    logger.info(f"📋 SOLUTION OVERVIEW:   Total solutions: {len(workspace.solutions)}")
    logger.info(f"📋 SOLUTION OVERVIEW:   Verified solutions: {len(verified_solutions)}")
    logger.info(f"📋 SOLUTION OVERVIEW:   Total characters: {total_chars:,}")
    logger.info(f"📋 SOLUTION OVERVIEW:   Average length: {avg_chars:.0f} chars")

    # Individual solution details
    for i, solution in enumerate(workspace.solutions):
        status = "✅ VERIFIED" if solution.is_verified else "❌ UNVERIFIED"
        logger.info(f"📋 SOLUTION OVERVIEW: Solution {i+1} (Agent {solution.agent_id}):")
        logger.info(f"📋 SOLUTION OVERVIEW:   Status: {status}")
        logger.info(f"📋 SOLUTION OVERVIEW:   Length: {len(solution.solution):,} chars")
        logger.info(f"📋 SOLUTION OVERVIEW:   Confidence: {solution.confidence:.2f}")
        logger.info(f"📋 SOLUTION OVERVIEW:   Verification score: {solution.verification_score:.2f}")
        logger.info(f"📋 SOLUTION OVERVIEW:   Reasoning tokens: {solution.reasoning_tokens:,}")
        logger.info(f"📋 SOLUTION OVERVIEW:   Temperature: {solution.temperature}")

        # Show solution preview
        preview = solution.solution[:300].replace('\n', ' ').strip()
        if len(solution.solution) > 300:
            preview += "..."
        logger.info(f"📋 SOLUTION OVERVIEW:   Preview: {preview}")