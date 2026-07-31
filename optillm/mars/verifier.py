"""
MARS Verification system implementing 5-pass verification threshold with parallel execution
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from .workspace import MARSWorkspace, AgentSolution
from .agent import MARSAgent

logger = logging.getLogger(__name__)

class MARSVerifier:
    """Multi-pass verification system inspired by IMO25 solver"""

    def __init__(self, agents: List[MARSAgent], workspace: MARSWorkspace, config: Dict[str, Any]):
        self.agents = agents
        self.workspace = workspace
        self.config = config
        self.verification_threshold = config.get('verification_passes_required', 5)

    def verify_solutions(self, request_id: str = None) -> Dict[str, Any]:
        """Run comprehensive verification on all solutions in workspace"""
        logger.info(f"Starting verification process with {self.verification_threshold}-pass threshold")

        verification_summary = {
            'total_verifications': 0,
            'solutions_verified': 0,
            'consensus_reached': False,
            'verification_details': []
        }

        solutions = self.workspace.solutions
        if not solutions:
            logger.warning("No solutions to verify")
            return verification_summary

        for solution in solutions:
            solution_verification = self._verify_single_solution(solution, request_id)
            verification_summary['verification_details'].append(solution_verification)
            verification_summary['total_verifications'] += solution_verification['verification_count']

            if solution_verification['passes_threshold']:
                verification_summary['solutions_verified'] += 1

        # Check for consensus
        verified_solutions = self.workspace.get_verified_solutions()
        verification_summary['consensus_reached'] = len(verified_solutions) >= self.config.get('consensus_threshold', 2)

        logger.info(f"Verification complete: {verification_summary['solutions_verified']} solutions verified")
        return verification_summary

    async def verify_solutions_parallel(
        self,
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Dict[str, Any]:
        """Run comprehensive verification on all solutions in workspace with parallel execution"""
        logger.info(f"Starting parallel verification process with {self.verification_threshold}-pass threshold")

        verification_summary = {
            'total_verifications': 0,
            'solutions_verified': 0,
            'consensus_reached': False,
            'verification_details': []
        }

        solutions = self.workspace.solutions
        if not solutions:
            logger.warning("No solutions to verify")
            return verification_summary

        # Verify all solutions in parallel
        async def verify_solution_async(solution: AgentSolution):
            """Async wrapper for single solution verification"""
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(
                    executor,
                    self._verify_single_solution,
                    solution,
                    request_id
                )
                return result
            except Exception as e:
                logger.error(f"Verification failed for solution from agent {solution.agent_id}: {str(e)}")
                return {
                    'solution_agent_id': solution.agent_id,
                    'verification_count': 0,
                    'consecutive_passes': 0,
                    'passes_threshold': False,
                    'verification_results': []
                }

        # Run verifications in parallel
        tasks = [verify_solution_async(solution) for solution in solutions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Verification task failed: {str(result)}")
                continue

            verification_summary['verification_details'].append(result)
            verification_summary['total_verifications'] += result['verification_count']

            if result['passes_threshold']:
                verification_summary['solutions_verified'] += 1

        # Check for consensus
        verified_solutions = self.workspace.get_verified_solutions()
        verification_summary['consensus_reached'] = len(verified_solutions) >= self.config.get('consensus_threshold', 2)

        logger.info(f"Parallel verification complete: {verification_summary['solutions_verified']} solutions verified")
        return verification_summary

    def _verify_single_solution(self, solution: AgentSolution, request_id: str = None) -> Dict[str, Any]:
        """Verify a single solution with multiple passes"""
        logger.info(f"Verifying solution from agent {solution.agent_id}")

        verification_results = []
        consecutive_passes = 0
        max_verification_attempts = self.config.get('max_verification_attempts', 10)

        for attempt in range(max_verification_attempts):
            # Select a different agent for verification
            verifier_agent = self._select_verifier_agent(solution.agent_id)
            if not verifier_agent:
                logger.warning("No suitable verifier agent available")
                break

            try:
                # Perform verification
                verification = verifier_agent.verify_solution(
                    problem=self.workspace.problem,
                    solution=solution.solution,
                    verifier_id=verifier_agent.agent_id,
                    solution_agent_id=solution.agent_id,
                    request_id=request_id
                )

                verification_results.append(verification)
                self.workspace.add_verification(verification)

                # Track consecutive passes
                if verification.assessment == "CORRECT":
                    consecutive_passes += 1
                    logger.info(f"Verification pass {consecutive_passes}/{self.verification_threshold}")

                    # Check if we've reached the threshold
                    if consecutive_passes >= self.verification_threshold:
                        logger.info(f"Solution from agent {solution.agent_id} passed {self.verification_threshold}-pass verification")
                        break
                else:
                    consecutive_passes = 0  # Reset on failure
                    logger.info(f"Verification failed: {verification.assessment}")

            except Exception as e:
                logger.error(f"Verification attempt {attempt + 1} failed: {str(e)}")
                consecutive_passes = 0

        return {
            'solution_agent_id': solution.agent_id,
            'verification_count': len(verification_results),
            'consecutive_passes': consecutive_passes,
            'passes_threshold': consecutive_passes >= self.verification_threshold,
            'verification_results': [
                {
                    'verifier_id': v.verifier_id,
                    'assessment': v.assessment,
                    'confidence': v.confidence,
                    'issues_count': len(v.issues)
                }
                for v in verification_results
            ]
        }

    def _select_verifier_agent(self, solution_agent_id: int) -> MARSAgent:
        """Select an agent different from the solution creator for verification"""
        available_agents = [agent for agent in self.agents if agent.agent_id != solution_agent_id]
        if not available_agents:
            # If no other agents available, use any agent
            available_agents = self.agents

        # Prefer agents with different temperatures for diverse verification
        if len(available_agents) > 1:
            # Select agent with most different temperature
            solution_agent = next((a for a in self.agents if a.agent_id == solution_agent_id), None)
            if solution_agent:
                solution_temp = solution_agent.temperature
                available_agents.sort(key=lambda a: abs(a.temperature - solution_temp), reverse=True)

        return available_agents[0] if available_agents else None

    def iterative_improvement(self, request_id: str = None) -> Dict[str, Any]:
        """Run iterative improvement on solutions that failed verification"""
        logger.info("Starting iterative improvement process")

        improvement_summary = {
            'solutions_improved': 0,
            'improvement_attempts': 0,
            'total_reasoning_tokens': 0
        }

        # Get solutions that need improvement
        unverified_solutions = [s for s in self.workspace.solutions if not s.is_verified]

        for solution in unverified_solutions:
            if solution.verification_results:
                # Get the most recent verification feedback
                latest_verification = solution.verification_results[-1]

                if latest_verification['assessment'] in ['INCORRECT', 'INCOMPLETE']:
                    # Find the original agent to improve their solution
                    original_agent = next((a for a in self.agents if a.agent_id == solution.agent_id), None)

                    if original_agent:
                        try:
                            improved_solution, reasoning_tokens = original_agent.improve_solution(
                                problem=self.workspace.problem,
                                current_solution=solution.solution,
                                feedback=latest_verification['detailed_report'],
                                issues=latest_verification['issues'],
                                request_id=request_id
                            )

                            # Update solution with improvement
                            solution.solution = improved_solution
                            solution.timestamp = datetime.now()
                            solution.reasoning_tokens += reasoning_tokens

                            improvement_summary['solutions_improved'] += 1
                            improvement_summary['total_reasoning_tokens'] += reasoning_tokens

                            logger.info(f"Improved solution from agent {solution.agent_id}")

                        except Exception as e:
                            logger.error(f"Failed to improve solution from agent {solution.agent_id}: {str(e)}")

                    improvement_summary['improvement_attempts'] += 1

        return improvement_summary

    async def iterative_improvement_parallel(
        self,
        request_id: str = None,
        executor: ThreadPoolExecutor = None
    ) -> Dict[str, Any]:
        """Run iterative improvement on solutions that failed verification with parallel execution"""
        logger.info("Starting parallel iterative improvement process")

        improvement_summary = {
            'solutions_improved': 0,
            'improvement_attempts': 0,
            'total_reasoning_tokens': 0
        }

        # Get solutions that need improvement
        unverified_solutions = [s for s in self.workspace.solutions if not s.is_verified]

        # Filter solutions that have verification feedback and can be improved
        improvable_solutions = []
        for solution in unverified_solutions:
            if solution.verification_results:
                latest_verification = solution.verification_results[-1]
                if latest_verification['assessment'] in ['INCORRECT', 'INCOMPLETE']:
                    original_agent = next((a for a in self.agents if a.agent_id == solution.agent_id), None)
                    if original_agent:
                        improvable_solutions.append((solution, original_agent, latest_verification))

        if not improvable_solutions:
            logger.info("No solutions need improvement")
            return improvement_summary

        # Improve solutions in parallel
        async def improve_solution_async(solution_data):
            """Async wrapper for solution improvement"""
            solution, agent, verification = solution_data
            loop = asyncio.get_event_loop()

            try:
                improved_solution, reasoning_tokens = await loop.run_in_executor(
                    executor,
                    agent.improve_solution,
                    self.workspace.problem,
                    solution.solution,
                    verification['detailed_report'],
                    verification['issues'],
                    request_id
                )

                # Update solution with improvement
                solution.solution = improved_solution
                solution.timestamp = datetime.now()
                solution.reasoning_tokens += reasoning_tokens

                logger.info(f"Improved solution from agent {solution.agent_id}")
                return solution.agent_id, True, reasoning_tokens, None

            except Exception as e:
                logger.error(f"Failed to improve solution from agent {solution.agent_id}: {str(e)}")
                return solution.agent_id, False, 0, e

        # Run improvements in parallel
        tasks = [improve_solution_async(sol_data) for sol_data in improvable_solutions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            improvement_summary['improvement_attempts'] += 1

            if isinstance(result, Exception):
                logger.error(f"Improvement task failed: {str(result)}")
                continue

            agent_id, success, tokens, error = result
            if success:
                improvement_summary['solutions_improved'] += 1
                improvement_summary['total_reasoning_tokens'] += tokens

        logger.info(f"Parallel improvement complete: {improvement_summary['solutions_improved']} solutions improved")
        return improvement_summary

    def final_consensus_check(self) -> bool:
        """Final check to determine if consensus has been reached"""
        verified_solutions = self.workspace.get_verified_solutions()
        consensus_threshold = self.config.get('consensus_threshold', 2)

        has_consensus = len(verified_solutions) >= consensus_threshold

        if has_consensus:
            logger.info(f"Consensus reached with {len(verified_solutions)} verified solutions")

            # Log the consensus solutions for analysis
            for solution in verified_solutions:
                logger.info(f"Verified solution from agent {solution.agent_id} (score: {solution.verification_score:.2f})")
        else:
            logger.info(f"No consensus: only {len(verified_solutions)} solutions verified (need {consensus_threshold})")

        return has_consensus