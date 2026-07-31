"""
Uncertainty-Routed Chain-of-Thought Implementation

This module implements uncertainty-routed CoT that generates multiple reasoning samples,
evaluates confidence through consistency, and routes to either majority voting or greedy decoding.
"""

import re
import logging
from typing import List, Dict, Any
from collections import Counter
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class UncertaintyRoutedCoT:
    """
    Implements uncertainty-routed chain-of-thought reasoning.

    The approach:
    1. Generate k chain-of-thought samples
    2. Evaluate confidence through consistency analysis
    3. Route to majority vote (high confidence) or greedy sample (low confidence)
    """

    def __init__(self, client, model: str, max_tokens: int = 16382, request_config: Dict[str, Any] = None):
        self.client = client
        self.model = model
        # Read max_completion_tokens (preferred) or max_tokens (deprecated) from request_config
        if request_config:
            self.max_tokens = request_config.get('max_completion_tokens') or request_config.get('max_tokens', max_tokens)
        else:
            self.max_tokens = max_tokens
        self.completion_tokens = 0
    
    def generate_with_uncertainty_routing(
        self,
        prompt: str,
        num_samples: int = 3,
        confidence_threshold: float = 0.7,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> Dict[str, Any]:
        """
        Generate response using uncertainty-routed chain-of-thought.
        
        Args:
            prompt: The prompt to generate responses for
            num_samples: Number of samples to generate for uncertainty evaluation
            confidence_threshold: Threshold for routing decision
            temperature: Sampling temperature for multiple samples
            top_p: Top-p parameter for sampling
            
        Returns:
            Dict containing final response, confidence score, and routing decision
        """
        logger.info(f"Generating {num_samples} samples for uncertainty routing")
        
        # Generate multiple samples
        samples = self._generate_multiple_samples(
            prompt, num_samples, temperature, top_p
        )
        
        # Generate greedy sample for comparison
        greedy_sample = self._generate_greedy_sample(prompt)
        
        # Extract thinking and answers from samples
        sample_data = []
        for sample in samples:
            thinking = self._extract_thinking(sample)
            answer = self._extract_answer(sample)
            sample_data.append({
                "full_response": sample,
                "thinking": thinking,
                "answer": answer
            })
        
        greedy_thinking = self._extract_thinking(greedy_sample)
        greedy_answer = self._extract_answer(greedy_sample)
        
        # Evaluate confidence through consistency
        confidence_score = self._evaluate_confidence(sample_data)
        
        # Log confidence evaluation details
        logger.debug(f"Confidence evaluation completed: {confidence_score:.3f}")
        logger.debug(f"Sample answers: {[sample['answer'][:50] + '...' if len(sample['answer']) > 50 else sample['answer'] for sample in sample_data if sample['answer']]}")
        
        # Route decision based on confidence
        if confidence_score >= confidence_threshold:
            # High confidence: use majority vote
            final_response = self._majority_vote_response(sample_data)
            routing_decision = "majority_vote"
            logger.info(f"High confidence ({confidence_score:.3f} >= {confidence_threshold}) - using majority vote")
        else:
            # Low confidence: use greedy sample
            final_response = greedy_sample
            routing_decision = "greedy"
            logger.info(f"Low confidence ({confidence_score:.3f} < {confidence_threshold}) - using greedy sample")
        
        return {
            "final_response": final_response,
            "confidence_score": confidence_score,
            "routing_decision": routing_decision,
            "samples": sample_data,
            "greedy_sample": {
                "full_response": greedy_sample,
                "thinking": greedy_thinking,
                "answer": greedy_answer
            },
            "completion_tokens": self.completion_tokens
        }
    
    def _generate_multiple_samples(
        self, 
        prompt: str, 
        num_samples: int, 
        temperature: float, 
        top_p: float
    ) -> List[str]:
        """Generate multiple samples by calling the API multiple times."""
        samples = []
        
        for i in range(num_samples):
            logger.debug(f"Generating sample {i+1}/{num_samples}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            self.completion_tokens += response.usage.completion_tokens

            # Check for truncated or empty response
            if (response is None or
                not response.choices or
                response.choices[0].message.content is None or
                response.choices[0].finish_reason == "length"):
                logger.warning(f"Sample {i+1}/{num_samples} truncated or empty, using empty string")
                samples.append("")
            else:
                samples.append(response.choices[0].message.content.strip())
        
        return samples
    
    def _generate_greedy_sample(self, prompt: str) -> str:
        """Generate a single greedy sample with temperature=0."""
        logger.debug("Generating greedy sample")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.0  # Greedy decoding
        )

        self.completion_tokens += response.usage.completion_tokens

        # Check for truncated or empty response
        if (response is None or
            not response.choices or
            response.choices[0].message.content is None or
            response.choices[0].finish_reason == "length"):
            logger.error("Greedy sample truncated or empty. Consider increasing max_tokens.")
            return "Error: Response was truncated due to token limit. Please increase max_tokens or max_completion_tokens."

        return response.choices[0].message.content.strip()
    
    def _extract_thinking(self, response: str) -> str:
        """Extract content from <think> tags."""
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from the response."""
        # Look for answer after </think> tag
        think_end = response.find('</think>')
        if think_end != -1:
            answer_part = response[think_end + 8:].strip()
        else:
            answer_part = response.strip()
        
        # Try to extract final answer with common patterns
        patterns = [
            r'(?:the )?(?:final )?answer is:?\s*(.+?)(?:\n|$)',
            r'(?:therefore|thus|so),?\s*(?:the )?(?:answer is:?\s*)?(.+?)(?:\n|$)',
            r'(?:conclusion|result):?\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_part, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: return the first significant line after thinking
        lines = answer_part.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Skip very short lines
                return line
        
        return answer_part[:200] if answer_part else ""  # Truncate if too long
    
    def _evaluate_confidence(self, sample_data: List[Dict[str, Any]]) -> float:
        """
        Evaluate confidence based on consistency across samples.
        
        Returns a confidence score between 0 and 1.
        """
        if len(sample_data) < 2:
            return 0.5  # Neutral confidence for single sample
        
        # Extract answers and thinking for analysis
        answers = [sample["answer"] for sample in sample_data if sample["answer"]]
        thinking_texts = [sample["thinking"] for sample in sample_data if sample["thinking"]]
        
        if not answers:
            return 0.1  # Very low confidence if no answers extracted
        
        # Evaluate answer consistency
        answer_consistency = self._calculate_answer_consistency(answers)
        
        # Evaluate reasoning consistency  
        reasoning_consistency = self._calculate_reasoning_consistency(thinking_texts)
        
        # Combine metrics (weighted average)
        confidence = (0.6 * answer_consistency + 0.4 * reasoning_consistency)
        
        logger.debug(f"Answer consistency: {answer_consistency:.3f} (weight: 0.6)")
        logger.debug(f"Reasoning consistency: {reasoning_consistency:.3f} (weight: 0.4)")
        logger.debug(f"Combined confidence: {confidence:.3f}")
        
        # Log additional details for debugging low confidence
        if confidence < 0.5:
            logger.debug(f"Low confidence detected. Sample count: {len(sample_data)}")
            logger.debug(f"Answers found: {len(answers)}, Thinking texts: {len(thinking_texts)}")
            if answers:
                logger.debug(f"Sample answers: {answers}")
            if len(answers) >= 2:
                logger.debug(f"Most common answer appears {max(Counter(answers).values())} times out of {len(answers)}")
        
        return confidence
    
    def _calculate_answer_consistency(self, answers: List[str]) -> float:
        """Calculate consistency of final answers."""
        if len(answers) < 2:
            return 0.5
        
        # Normalize answers for comparison
        normalized_answers = []
        for answer in answers:
            # Remove common variations and normalize
            norm_answer = re.sub(r'[^\w\s]', '', answer.lower().strip())
            norm_answer = re.sub(r'\s+', ' ', norm_answer)
            normalized_answers.append(norm_answer)
        
        # Count occurrences
        answer_counts = Counter(normalized_answers)
        most_common_count = answer_counts.most_common(1)[0][1]
        total_answers = len(answers)
        
        # Calculate agreement ratio
        agreement_ratio = most_common_count / total_answers
        
        logger.debug(f"Answer distribution: {dict(answer_counts)}")
        logger.debug(f"Agreement ratio: {agreement_ratio:.3f} ({most_common_count}/{total_answers})")
        
        # Also consider semantic similarity for non-identical answers
        max_similarity = 0.0
        for i, ans1 in enumerate(normalized_answers):
            for j, ans2 in enumerate(normalized_answers[i+1:], i+1):
                similarity = SequenceMatcher(None, ans1, ans2).ratio()
                max_similarity = max(max_similarity, similarity)
        
        # Combine exact matches and semantic similarity
        consistency = max(agreement_ratio, max_similarity)
        
        return min(consistency, 1.0)
    
    def _calculate_reasoning_consistency(self, thinking_texts: List[str]) -> float:
        """Calculate consistency of reasoning processes."""
        if len(thinking_texts) < 2:
            return 0.5
        
        # Calculate pairwise similarity of reasoning
        similarities = []
        for i, text1 in enumerate(thinking_texts):
            for j, text2 in enumerate(thinking_texts[i+1:], i+1):
                # Use sequence matcher for text similarity
                similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Return average similarity
        avg_similarity = sum(similarities) / len(similarities)
        
        logger.debug(f"Reasoning similarity pairs: {[f'{s:.3f}' for s in similarities]}")
        logger.debug(f"Average reasoning similarity: {avg_similarity:.3f}")
        
        return min(avg_similarity, 1.0)
    
    def _majority_vote_response(self, sample_data: List[Dict[str, Any]]) -> str:
        """
        Create response based on majority vote of answers and best reasoning.
        """
        # Get most common answer
        answers = [sample["answer"] for sample in sample_data if sample["answer"]]
        if not answers:
            return sample_data[0]["full_response"]
        
        # Normalize and count answers
        normalized_answers = []
        for answer in answers:
            norm_answer = re.sub(r'[^\w\s]', '', answer.lower().strip())
            norm_answer = re.sub(r'\s+', ' ', norm_answer)
            normalized_answers.append(norm_answer)
        
        answer_counts = Counter(normalized_answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Find the sample with the most common answer and best reasoning
        best_sample = None
        best_reasoning_length = 0
        
        for i, sample in enumerate(sample_data):
            if sample["answer"]:
                norm_answer = re.sub(r'[^\w\s]', '', sample["answer"].lower().strip())
                norm_answer = re.sub(r'\s+', ' ', norm_answer)
                
                if norm_answer == most_common_answer:
                    reasoning_length = len(sample["thinking"])
                    if reasoning_length > best_reasoning_length:
                        best_reasoning_length = reasoning_length
                        best_sample = sample
        
        if best_sample:
            return best_sample["full_response"]
        else:
            # Fallback to first sample with the most common answer
            for sample in sample_data:
                if sample["answer"]:
                    norm_answer = re.sub(r'[^\w\s]', '', sample["answer"].lower().strip())
                    norm_answer = re.sub(r'\s+', ' ', norm_answer)
                    if norm_answer == most_common_answer:
                        return sample["full_response"]
        
        # Final fallback
        return sample_data[0]["full_response"]
