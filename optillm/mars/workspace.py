"""
Shared workspace for MARS agent collaboration
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentSolution:
    """Represents a solution attempt by an agent"""
    agent_id: str  # Changed to str to support aggregated agent IDs like "agg_123456"
    solution: str
    confidence: float
    reasoning_tokens: int
    total_tokens: int
    solution_length: int
    is_verified: bool = False
    verification_score: float = 0.0
    temperature: float = 0.7  # Default temperature
    timestamp: datetime = field(default_factory=datetime.now)
    verification_results: List[Dict] = field(default_factory=list)

@dataclass
class VerificationResult:
    """Represents a verification attempt result"""
    verifier_id: int
    solution_id: str
    assessment: str  # CORRECT, INCORRECT, INCOMPLETE
    confidence: float
    issues: List[str]
    suggestions: List[str]
    detailed_report: str
    timestamp: datetime

class MARSWorkspace:
    """Shared workspace for agent collaboration and solution tracking"""

    def __init__(self, problem: str, config: Dict[str, Any]):
        self.problem = problem
        self.config = config
        self.solutions: List[AgentSolution] = []
        self.verification_results: List[VerificationResult] = []
        self.synthesis_attempts: List[Dict] = []
        self.final_solution: Optional[str] = None
        self.iteration_count = 0
        self.total_reasoning_tokens = 0

        logger.info(f"Initialized MARS workspace for problem: {problem[:100]}...")

    def add_solution(self, agent_solution: AgentSolution) -> str:
        """Add a new agent solution to the workspace"""
        # Keep the original agent_id, don't overwrite it
        solution_id = f"agent_{agent_solution.agent_id}_iter_{self.iteration_count}"
        self.solutions.append(agent_solution)
        self.total_reasoning_tokens += agent_solution.reasoning_tokens

        logger.info(f"Added solution {solution_id} with {agent_solution.reasoning_tokens} reasoning tokens")
        return solution_id

    def add_verification(self, verification: VerificationResult):
        """Add a verification result to the workspace"""
        self.verification_results.append(verification)

        # Update the corresponding solution's verification status
        # Extract agent_id from solution_id (format: "agent_X_iter_Y")
        if verification.solution_id.startswith("agent_"):
            try:
                agent_id_str = verification.solution_id.removeprefix("agent_").rsplit("_iter_", 1)[0]

                # Handle both integer and string agent_ids for backward compatibility
                for solution in self.solutions:
                    if str(solution.agent_id) == agent_id_str:
                        solution.verification_results.append({
                            'assessment': verification.assessment,
                            'confidence': verification.confidence,
                            'issues': verification.issues,
                            'detailed_report': verification.detailed_report
                        })

                        # Update verification score (average of all verifications)
                        verified_count = len([v for v in solution.verification_results if v['assessment'] == 'CORRECT'])
                        total_verifications = len(solution.verification_results)
                        solution.verification_score = verified_count / total_verifications if total_verifications > 0 else 0

                        # Use count-based verification instead of percentage
                        consecutive_correct = 0
                        for v in reversed(solution.verification_results):
                            if v['assessment'] == 'CORRECT':
                                consecutive_correct += 1
                            else:
                                break

                        verification_threshold = self.config.get('verification_passes_required', 5)
                        solution.is_verified = consecutive_correct >= verification_threshold
                        break
            except (IndexError, ValueError):
                logger.warning(f"Invalid solution_id format: {verification.solution_id}")

        logger.info(f"Added verification for {verification.solution_id}: {verification.assessment}")

    def get_verified_solutions(self) -> List[AgentSolution]:
        """Get all solutions that have passed verification"""
        return [s for s in self.solutions if s.is_verified]

    def get_best_solution(self) -> Optional[AgentSolution]:
        """Get the best solution based on verification score and confidence"""
        if not self.solutions:
            return None

        verified_solutions = self.get_verified_solutions()
        if verified_solutions:
            # Among verified solutions, pick the one with highest confidence
            return max(verified_solutions, key=lambda s: s.confidence)
        else:
            # If no verified solutions, pick the one with highest verification score
            return max(self.solutions, key=lambda s: s.verification_score)

    def has_consensus(self) -> bool:
        """Check if we have enough verified solutions to reach consensus"""
        verified_count = len(self.get_verified_solutions())
        required_consensus = self.config.get('consensus_threshold', 2)
        return verified_count >= required_consensus

    def should_continue_iteration(self) -> bool:
        """Determine if we should continue with another iteration"""
        max_iterations = self.config.get('max_iterations', 5)
        min_verified = self.config.get('min_verified_solutions', 1)

        # Continue if we haven't reached max iterations and don't have enough verified solutions
        return (self.iteration_count < max_iterations and
                len(self.get_verified_solutions()) < min_verified)

    def get_synthesis_input(self) -> Dict[str, Any]:
        """Prepare input data for solution synthesis"""
        return {
            'problem': self.problem,
            'solutions': [
                {
                    'agent_id': s.agent_id,
                    'solution': s.solution,
                    'confidence': s.confidence,
                    'verification_score': s.verification_score,
                    'verification_results': s.verification_results
                }
                for s in self.solutions
            ],
            'verification_summary': self._get_verification_summary(),
            'total_reasoning_tokens': self.total_reasoning_tokens
        }

    def _get_verification_summary(self) -> Dict[str, Any]:
        """Generate a summary of all verification results"""
        total_verifications = len(self.verification_results)
        if total_verifications == 0:
            return {'total': 0, 'correct': 0, 'incorrect': 0, 'incomplete': 0}

        assessments = [v.assessment for v in self.verification_results]
        return {
            'total': total_verifications,
            'correct': assessments.count('CORRECT'),
            'incorrect': assessments.count('INCORRECT'),
            'incomplete': assessments.count('INCOMPLETE'),
            'avg_confidence': sum(v.confidence for v in self.verification_results) / total_verifications
        }

    def set_final_solution(self, solution: str):
        """Set the final synthesized solution"""
        self.final_solution = solution
        logger.info("Final solution set in workspace")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the workspace state"""
        return {
            'problem': self.problem,
            'total_solutions': len(self.solutions),
            'verified_solutions': len(self.get_verified_solutions()),
            'total_verifications': len(self.verification_results),
            'iterations_completed': self.iteration_count,
            'total_reasoning_tokens': self.total_reasoning_tokens,
            'has_consensus': self.has_consensus(),
            'final_solution': self.final_solution,
            'verification_summary': self._get_verification_summary()
        }
