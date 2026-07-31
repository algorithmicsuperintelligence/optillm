"""
MARS Agent implementation with OpenRouter reasoning API
"""

import logging
from typing import Dict, Any, Tuple
from datetime import datetime
from .prompts import (
    MATHEMATICAL_SYSTEM_PROMPT,
    AGENT_EXPLORATION_PROMPT,
    VERIFICATION_PROMPT,
    IMPROVEMENT_PROMPT
)
from .workspace import AgentSolution, VerificationResult

logger = logging.getLogger(__name__)

class MARSAgent:
    """Individual agent for mathematical reasoning with OpenRouter reasoning API"""

    def __init__(self, agent_id: int, client, model: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.client = client
        self.model = model
        self.config = config
        self.temperature = self._assign_temperature()

    def _assign_temperature(self) -> float:
        """Assign temperature based on agent ID for 3-agent configuration"""
        temperatures = [0.3, 0.6, 1.0]  # Low, Medium, High reasoning effort
        return temperatures[self.agent_id % len(temperatures)]

    def _get_reasoning_effort(self) -> str:
        """Get reasoning effort level based on agent temperature"""
        if self.temperature <= 0.4:
            return "low"  # ~20% of max_tokens for reasoning
        elif self.temperature <= 0.8:
            return "medium"  # ~50% of max_tokens for reasoning
        else:
            return "high"  # ~80% of max_tokens for reasoning

    def generate_solution(self, problem: str, request_id: str = None) -> Tuple[AgentSolution, int]:
        """Generate a solution for the given problem using reasoning API"""
        import time
        start_time = time.time()
        logger.info(f"🤖 AGENT {self.agent_id}: Starting solution generation (temp: {self.temperature}, effort: {self._get_reasoning_effort()})")
        logger.info(f"🤖 AGENT {self.agent_id}: Problem length: {len(problem)} characters")

        # Prepare the prompt
        exploration_prompt = AGENT_EXPLORATION_PROMPT.format(
            agent_id=self.agent_id,
            temperature=self.temperature,
            problem=problem
        )

        # Configure reasoning parameters - simplified with effort only
        reasoning_effort = self._get_reasoning_effort()
        max_tokens = self.config['max_tokens']
        logger.info(f"🤖 AGENT {self.agent_id}: Using max_tokens={max_tokens}, reasoning_effort={reasoning_effort}")

        reasoning_config = {
            "effort": reasoning_effort
        }

        try:
            # Make API call with reasoning via extra_body for OpenRouter compatibility
            api_start = time.time()
            logger.info(f"🤖 AGENT {self.agent_id}: Making API call to {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MATHEMATICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": exploration_prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=300,  # 5 minute timeout for complex problems
                extra_body={
                    "reasoning": reasoning_config
                }
            )
            api_duration = time.time() - api_start
            logger.info(f"🤖 AGENT {self.agent_id}: API call completed in {api_duration:.2f}s")

            solution_text = response.choices[0].message.content.strip()

            # ENHANCED LOGGING: Log solution details
            solution_length = len(solution_text)
            word_count = len(solution_text.split())
            has_boxed = "\\boxed{" in solution_text
            has_proof_words = any(word in solution_text.lower() for word in ['therefore', 'thus', 'proof', 'qed'])

            logger.info(f"🤖 AGENT {self.agent_id}: Solution analysis:")
            logger.info(f"  📝 Length: {solution_length:,} chars, {word_count:,} words")
            logger.info(f"  📦 Has boxed answer: {has_boxed}")
            logger.info(f"  🔍 Has proof indicators: {has_proof_words}")
            logger.info(f"  📄 Preview: {solution_text[:200]}{'...' if len(solution_text) > 200 else ''}")
            logger.info(f"  📄 Last 100 chars: ...{solution_text[-100:] if solution_length > 100 else solution_text}")

            # Extract reasoning tokens from the correct nested structure
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

            reasoning_ratio = (reasoning_tokens / total_tokens * 100) if total_tokens > 0 else 0
            logger.info(f"🤖 AGENT {self.agent_id}: Token usage: reasoning={reasoning_tokens:,}, total={total_tokens:,} ({reasoning_ratio:.1f}% reasoning)")

            # Extract confidence from solution (heuristic based on response characteristics)
            confidence = self._estimate_confidence(solution_text)
            logger.info(f"🤖 AGENT {self.agent_id}: Estimated confidence: {confidence:.3f}")

            # Create agent solution object with enhanced metadata
            agent_solution = AgentSolution(
                agent_id=str(self.agent_id),  # Convert to str for compatibility
                solution=solution_text,
                confidence=confidence,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                solution_length=solution_length,
                temperature=self.temperature
            )

            total_duration = time.time() - start_time
            logger.info(f"🤖 AGENT {self.agent_id}: ✅ Solution generated in {total_duration:.2f}s (API: {api_duration:.2f}s, processing: {total_duration-api_duration:.2f}s)")
            return agent_solution, reasoning_tokens

        except Exception as e:
            error_duration = time.time() - start_time
            logger.error(f"🤖 AGENT {self.agent_id}: ❌ Error generating solution after {error_duration:.2f}s: {str(e)}")
            logger.error(f"🤖 AGENT {self.agent_id}: Model: {self.model}, Temperature: {self.temperature}, Max tokens: {max_tokens}")
            # Return empty solution with error indication
            error_message = f"Error generating solution: {str(e)}"
            error_solution = AgentSolution(
                agent_id=str(self.agent_id),  # Convert to str for compatibility
                solution=error_message,
                confidence=0.0,
                reasoning_tokens=0,
                total_tokens=0,
                solution_length=len(error_message),
                temperature=self.temperature
            )
            return error_solution, 0

    def verify_solution(self, problem: str, solution: str, verifier_id: int, solution_agent_id: int, request_id: str = None) -> VerificationResult:
        """Verify a solution using mathematical reasoning"""
        import time
        start_time = time.time()
        logger.info(f"🔍 VERIFIER {self.agent_id}: Starting verification (target: Agent {solution_agent_id}, verifier_id: {verifier_id})")
        logger.info(f"🔍 VERIFIER {self.agent_id}: Solution length: {len(solution):,} chars")

        verification_prompt = VERIFICATION_PROMPT.format(
            problem=problem,
            solution=solution
        )

        # Use simplified verification with effort parameter
        max_tokens = self.config['max_tokens']

        try:
            api_start = time.time()
            logger.info(f"🔍 VERIFIER {self.agent_id}: Making verification API call...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MATHEMATICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": verification_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent verification
                timeout=180,
                extra_body={
                    "reasoning": {
                        "effort": "low"  # Low effort for verification consistency
                    }
                }
            )
            api_duration = time.time() - api_start
            logger.info(f"🔍 VERIFIER {self.agent_id}: Verification API call completed in {api_duration:.2f}s")

            verification_text = response.choices[0].message.content.strip()

            # Parse verification result
            assessment, confidence, issues, suggestions = self._parse_verification(verification_text)
            logger.info(f"🔍 VERIFIER {self.agent_id}: Assessment: {assessment}, Confidence: {confidence:.3f}")
            logger.info(f"🔍 VERIFIER {self.agent_id}: Issues found: {len(issues)}, Suggestions: {len(suggestions)}")
            if issues:
                logger.info(f"🔍 VERIFIER {self.agent_id}: Key issues: {issues[:2]}")

            result = VerificationResult(
                verifier_id=verifier_id,
                solution_id=f"agent_{solution_agent_id}_iter_0",  # Use the solution's agent_id
                assessment=assessment,
                confidence=confidence,
                issues=issues,
                suggestions=suggestions,
                detailed_report=verification_text,
                timestamp=datetime.now()
            )

            total_duration = time.time() - start_time
            logger.info(f"🔍 VERIFIER {self.agent_id}: ✅ Verification completed in {total_duration:.2f}s")
            return result

        except Exception as e:
            error_duration = time.time() - start_time
            logger.error(f"🔍 VERIFIER {self.agent_id}: ❌ Verification error after {error_duration:.2f}s: {str(e)}")
            return VerificationResult(
                verifier_id=verifier_id,
                solution_id=f"agent_{solution_agent_id}_iter_0",
                assessment="INCOMPLETE",
                confidence=0.0,
                issues=[f"Verification error: {str(e)}"],
                suggestions=["Retry verification"],
                detailed_report=f"Error during verification: {str(e)}",
                timestamp=datetime.now()
            )

    def improve_solution(self, problem: str, current_solution: str, feedback: str, issues: list, request_id: str = None) -> Tuple[str, int]:
        """Improve a solution based on verification feedback"""
        import time
        start_time = time.time()
        logger.info(f"🔧 IMPROVER {self.agent_id}: Starting solution improvement")
        logger.info(f"🔧 IMPROVER {self.agent_id}: Current solution: {len(current_solution):,} chars")
        logger.info(f"🔧 IMPROVER {self.agent_id}: Issues to address: {len(issues)}")

        improvement_prompt = IMPROVEMENT_PROMPT.format(
            problem=problem,
            current_solution=current_solution,
            feedback=feedback,
            issues="\n".join(f"- {issue}" for issue in issues)
        )

        # Use simplified improvement with high effort
        max_tokens = self.config['max_tokens']

        try:
            api_start = time.time()
            logger.info(f"🔧 IMPROVER {self.agent_id}: Making improvement API call...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MATHEMATICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": improvement_prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.temperature * 0.8,  # Slightly lower temperature for improvement
                timeout=300,
                extra_body={
                    "reasoning": {
                        "effort": "high"  # High effort for improvements
                    }
                }
            )
            api_duration = time.time() - api_start
            logger.info(f"🔧 IMPROVER {self.agent_id}: Improvement API call completed in {api_duration:.2f}s")

            improved_solution = response.choices[0].message.content.strip()
            reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)

            # Log improvement analysis
            length_change = len(improved_solution) - len(current_solution)
            logger.info(f"🔧 IMPROVER {self.agent_id}: Solution length change: {length_change:+,} chars")
            logger.info(f"🔧 IMPROVER {self.agent_id}: Improved solution preview: {improved_solution[:200]}{'...' if len(improved_solution) > 200 else ''}")

            total_duration = time.time() - start_time
            logger.info(f"🔧 IMPROVER {self.agent_id}: ✅ Solution improved in {total_duration:.2f}s with {reasoning_tokens:,} reasoning tokens")
            return improved_solution, reasoning_tokens

        except Exception as e:
            error_duration = time.time() - start_time
            logger.error(f"🔧 IMPROVER {self.agent_id}: ❌ Improvement error after {error_duration:.2f}s: {str(e)}")
            logger.warning(f"🔧 IMPROVER {self.agent_id}: Returning original solution due to error")
            return current_solution, 0  # Return original solution if improvement fails

    def _estimate_confidence(self, solution: str) -> float:
        """Estimate confidence based on solution characteristics"""
        confidence = 0.5  # Base confidence
        confidence_factors = []

        # Check for mathematical rigor indicators
        if "\\boxed{" in solution:
            confidence += 0.2
            confidence_factors.append("boxed_answer")
        if "therefore" in solution.lower() or "thus" in solution.lower():
            confidence += 0.1
            confidence_factors.append("logical_connectors")
        if "proof" in solution.lower():
            confidence += 0.1
            confidence_factors.append("proof_structure")
        if len(solution.split()) > 200:  # Detailed solutions tend to be more confident
            confidence += 0.1
            confidence_factors.append("detailed_solution")
        if "let" in solution.lower() and "assume" in solution.lower():
            confidence += 0.1
            confidence_factors.append("formal_approach")

        # Check for uncertainty indicators
        uncertainty_factors = []
        if "might" in solution.lower() or "possibly" in solution.lower():
            confidence -= 0.1
            uncertainty_factors.append("hedging_language")
        if "unsure" in solution.lower() or "not sure" in solution.lower():
            confidence -= 0.2
            uncertainty_factors.append("explicit_uncertainty")

        final_confidence = max(0.1, min(1.0, confidence))
        logger.debug(f"🤖 AGENT {self.agent_id}: Confidence factors: +{confidence_factors}, -{uncertainty_factors} → {final_confidence:.3f}")
        return final_confidence

    def _parse_verification(self, verification_text: str) -> Tuple[str, float, list, list]:
        """Parse verification result to extract structured information"""
        assessment = "INCOMPLETE"  # Default
        confidence = 0.5
        issues = []
        suggestions = []

        text_lower = verification_text.lower()

        # Determine assessment
        if "correct" in text_lower and "incorrect" not in text_lower:
            assessment = "CORRECT"
            confidence = 0.8
        elif "incorrect" in text_lower:
            assessment = "INCORRECT"
            confidence = 0.8
        elif "incomplete" in text_lower:
            assessment = "INCOMPLETE"
            confidence = 0.6

        # Extract confidence if explicitly mentioned
        import re
        confidence_match = re.search(r'confidence.*?(\d+).*?(?:out of|/)\s*(\d+)', text_lower)
        if confidence_match:
            conf_score = float(confidence_match.group(1))
            conf_total = float(confidence_match.group(2))
            confidence = conf_score / conf_total

        # Extract issues (simple heuristic)
        lines = verification_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['error', 'mistake', 'incorrect', 'wrong', 'issue']):
                issues.append(line.strip())

        # Extract suggestions (simple heuristic)
        for line in lines:
            line_lower = line.lower()
            if any(word in line_lower for word in ['suggest', 'recommend', 'should', 'could improve']):
                suggestions.append(line.strip())

        return assessment, confidence, issues, suggestions