"""
MARS Aggregator: RSA-inspired solution aggregation system
Implements recursive self-aggregation for combining and refining solutions
"""

import asyncio
import logging
import random
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from optillm import conversation_logger
from .workspace import MARSWorkspace, AgentSolution
from .prompts import SINGLE_REFINEMENT_PROMPT, MULTI_AGGREGATION_PROMPT

logger = logging.getLogger(__name__)


class MARSAggregator:
    """
    RSA-inspired aggregation system for combining solutions

    Key features:
    - Population management (N > K for diversity)
    - Recursive aggregation loops
    - Parallel execution of aggregation calls
    - Solution quality tracking
    """

    def __init__(self, client, model: str, config: Dict[str, Any]):
        self.client = client
        self.model = model
        self.config = config
        self.population_size = config.get('population_size', 6)
        self.aggregation_size = config.get('aggregation_size', 3)
        self.aggregation_loops = config.get('aggregation_loops', 3)
        self.max_tokens = config.get('max_tokens', 30000)

    async def run_aggregation_loops(
        self,
        workspace: MARSWorkspace,
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Run T iterations of RSA-style aggregation

        Args:
            workspace: MARS workspace containing solutions
            request_id: Request ID for logging
            executor: Thread pool for parallel execution

        Returns:
            Tuple of (total_reasoning_tokens, aggregation_summary)
        """
        logger.info(f"Starting {self.aggregation_loops} aggregation loops")

        total_reasoning_tokens = 0
        aggregation_history = []

        # Ensure we have enough solutions for aggregation
        self._ensure_population_size(workspace)

        for loop_idx in range(self.aggregation_loops):
            logger.info(f"Aggregation loop {loop_idx + 1}/{self.aggregation_loops}")

            # Run single aggregation loop
            loop_tokens, loop_summary = await self._run_single_aggregation_loop(
                workspace, loop_idx, request_id, executor
            )

            total_reasoning_tokens += loop_tokens
            aggregation_history.append({
                'loop': loop_idx,
                'tokens': loop_tokens,
                'summary': loop_summary
            })

            # Log progress
            logger.info(f"Loop {loop_idx + 1} complete: {loop_summary['solutions_generated']} new solutions")

        summary = {
            'total_loops': self.aggregation_loops,
            'total_reasoning_tokens': total_reasoning_tokens,
            'final_population_size': len(workspace.solutions),
            'aggregation_history': aggregation_history
        }

        logger.info(f"Aggregation complete: {summary['final_population_size']} solutions in final population")
        return total_reasoning_tokens, summary

    async def _run_single_aggregation_loop(
        self,
        workspace: MARSWorkspace,
        loop_idx: int,
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Run a single aggregation loop: sample K -> aggregate -> update population"""

        # Sample K solutions from current population
        sampled_solutions = self._sample_solutions_for_aggregation(workspace)

        # Generate M new solutions by aggregating sampled ones
        new_solutions, total_tokens = await self._generate_aggregated_solutions(
            workspace, sampled_solutions, request_id, executor
        )

        # Update population with new solutions
        self._update_population(workspace, new_solutions)

        loop_summary = {
            'sampled_solutions': len(sampled_solutions),
            'solutions_generated': len(new_solutions),
            'population_size': len(workspace.solutions),
            'total_tokens': total_tokens
        }

        return total_tokens, loop_summary

    def _sample_solutions_for_aggregation(self, workspace: MARSWorkspace) -> List[List[AgentSolution]]:
        """
        Sample K solutions from population for aggregation
        Uses different strategies for each sample to maintain diversity
        """
        all_solutions = workspace.solutions

        if len(all_solutions) < self.aggregation_size:
            # Not enough solutions, use what we have
            return [all_solutions]

        # Generate multiple samples for parallel aggregation
        samples = []
        num_samples = min(self.population_size // self.aggregation_size, 3)  # Max 3 parallel aggregations

        for i in range(num_samples):
            if i == 0:
                # First sample: best solutions by verification score
                sample = sorted(all_solutions, key=lambda s: s.verification_score, reverse=True)[:self.aggregation_size]
            elif i == 1:
                # Second sample: diverse solutions (by agent_id)
                by_agent = {}
                for sol in all_solutions:
                    if sol.agent_id not in by_agent:
                        by_agent[sol.agent_id] = []
                    by_agent[sol.agent_id].append(sol)

                sample = []
                for agent_solutions in by_agent.values():
                    if len(sample) < self.aggregation_size:
                        sample.append(max(agent_solutions, key=lambda s: s.confidence))
                    if len(sample) >= self.aggregation_size:
                        break

                # Fill remaining slots with best overall
                if len(sample) < self.aggregation_size:
                    remaining = [s for s in all_solutions if s not in sample]
                    sample.extend(sorted(remaining, key=lambda s: s.verification_score, reverse=True)[:self.aggregation_size - len(sample)])
            else:
                # Random sample for exploration
                sample = random.sample(all_solutions, min(self.aggregation_size, len(all_solutions)))

            samples.append(sample)

        logger.info(f"Generated {len(samples)} sample groups for aggregation")
        return samples

    async def _generate_aggregated_solutions(
        self,
        workspace: MARSWorkspace,
        sampled_solution_groups: List[List[AgentSolution]],
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Tuple[List[AgentSolution], int]:
        """Generate new solutions by aggregating sampled solutions in parallel"""

        async def aggregate_solution_group(solutions: List[AgentSolution]) -> Tuple[Optional[AgentSolution], int]:
            """Aggregate a single group of solutions"""
            loop = asyncio.get_event_loop()

            try:
                # Choose aggregation strategy based on number of solutions
                if len(solutions) == 1:
                    # Single solution refinement
                    prompt = SINGLE_REFINEMENT_PROMPT.format(
                        problem=workspace.problem,
                        candidate_solution=solutions[0].solution
                    )
                else:
                    # Multi-solution aggregation
                    candidate_text = ""
                    for i, sol in enumerate(solutions):
                        candidate_text += f"Solution {i+1} (Agent {sol.agent_id}, confidence: {sol.confidence:.2f}):\n"
                        candidate_text += sol.solution + "\n\n"

                    prompt = MULTI_AGGREGATION_PROMPT.format(
                        problem=workspace.problem,
                        candidate_solutions=candidate_text
                    )

                # Generate aggregated solution
                solution, tokens = await loop.run_in_executor(
                    executor,
                    self._call_model_for_aggregation,
                    prompt,
                    request_id
                )

                if solution:
                    # Create new AgentSolution with aggregated content
                    aggregated_solution = AgentSolution(
                        agent_id=f"agg_{datetime.now().strftime('%H%M%S')}",
                        solution=solution,
                        confidence=0.8,  # Base confidence for aggregated solutions
                        reasoning_tokens=tokens,
                        total_tokens=tokens,
                        solution_length=len(solution),
                        is_verified=False,
                        verification_score=0.0
                    )
                    return aggregated_solution, tokens

                return None, tokens

            except Exception as e:
                logger.error(f"Aggregation failed: {str(e)}")
                return None, 0

        # Run aggregations in parallel
        tasks = [aggregate_solution_group(group) for group in sampled_solution_groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_solutions = []
        total_tokens = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Aggregation task failed: {str(result)}")
                continue

            solution, tokens = result
            if solution:
                new_solutions.append(solution)
            total_tokens += tokens

        logger.info(f"Generated {len(new_solutions)} aggregated solutions with {total_tokens} reasoning tokens")
        return new_solutions, total_tokens

    def _call_model_for_aggregation(self, prompt: str, request_id: str = None) -> Tuple[str, int]:
        """Call the model to perform aggregation (synchronous for executor)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a mathematical reasoning expert focused on solution aggregation and refinement."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.7,  # Slightly higher temperature for creativity in aggregation
                timeout=300,
                extra_body={
                    "reasoning": {
                        "effort": "high"
                    }
                }
            )

            # Log provider call if conversation logging is enabled
            if request_id:
                provider_request = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a mathematical reasoning expert focused on solution aggregation and refinement."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7,
                    "extra_body": {
                        "reasoning": {
                            "effort": "high"
                        }
                    }
                }
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
                conversation_logger.log_provider_call(request_id, provider_request, response_dict)

            solution = response.choices[0].message.content.strip()

            # Extract reasoning tokens using correct nested structure (matching agent.py fix)
            reasoning_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                # Check completion_tokens_details first (OpenRouter structure)
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)
                # Fallback to direct usage field (standard OpenAI structure)
                if reasoning_tokens == 0:
                    reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)

            return solution, reasoning_tokens

        except Exception as e:
            logger.error(f"Model call for aggregation failed: {str(e)}")
            return "", 0

    def _update_population(self, workspace: MARSWorkspace, new_solutions: List[AgentSolution]) -> None:
        """Update population with new solutions, maintaining population size limit"""

        # Add new solutions to workspace
        for solution in new_solutions:
            workspace.add_solution(solution)

        # Maintain population size limit (N = population_size)
        all_solutions = workspace.solutions
        if len(all_solutions) > self.population_size:
            # Keep best solutions by verification score, then confidence
            sorted_solutions = sorted(
                all_solutions,
                key=lambda s: (s.verification_score, s.confidence),
                reverse=True
            )
            workspace.solutions = sorted_solutions[:self.population_size]

            logger.info(f"Population trimmed to {self.population_size} best solutions")

    def _ensure_population_size(self, workspace: MARSWorkspace) -> None:
        """Ensure we have minimum population size for effective aggregation"""
        current_size = len(workspace.solutions)

        if current_size < self.aggregation_size:
            logger.warning(f"Population size ({current_size}) < aggregation size ({self.aggregation_size})")
            logger.warning("Aggregation may be less effective with limited diversity")

        logger.info(f"Population ready: {current_size} solutions available for aggregation")
