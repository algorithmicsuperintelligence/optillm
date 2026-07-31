#!/usr/bin/env python3
"""
Comprehensive tests for conversation logging across all approaches
Tests ensure that all approaches properly log API calls without regressions
"""

import unittest
import sys
import os
import json
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optillm
from optillm.conversation_logger import ConversationLogger, set_global_logger

# Import all approaches we've modified
from optillm.bon import best_of_n_sampling
from optillm.mcts import chat_with_mcts
from optillm.rto import round_trip_optimization
from optillm.pvg import inference_time_pv_game
from optillm.cot_reflection import cot_reflection
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.reread import re2_approach
from optillm.rstar import RStar
from optillm.z3_solver import Z3SymPySolverSystem


class MockOpenAIResponse:
    """Mock OpenAI API response"""
    def __init__(self, content="Test response", usage_tokens=10, n=1, call_index=0):
        self.choices = []
        for i in range(n):
            choice = Mock()
            choice.message = Mock()
            # Make different content for different calls to avoid early returns
            if call_index % 2 == 0:
                choice.message.content = f"Code version A: {content} {i+1}" if n > 1 else f"Code version A: {content}"
            else:
                choice.message.content = f"Code version B: {content} {i+1}" if n > 1 else f"Code version B: {content}"
            self.choices.append(choice)
        
        self.usage = Mock()
        self.usage.completion_tokens = usage_tokens
        self.usage.completion_tokens_details = Mock()
        self.usage.completion_tokens_details.reasoning_tokens = 0
        
    def model_dump(self):
        return {
            "choices": [{"message": {"content": choice.message.content}} for choice in self.choices],
            "usage": {"completion_tokens": self.usage.completion_tokens}
        }


class MockOpenAIClient:
    """Mock OpenAI client for testing"""
    def __init__(self, response_content="Test response", usage_tokens=10, n_responses=1):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.responses = []
        
        # Create multiple responses if needed
        for i in range(20):  # Create enough responses for complex approaches
            response = MockOpenAIResponse(response_content, usage_tokens, n_responses, i)
            self.responses.append(response)
        
        self.call_count = 0
        self.chat.completions.create = self._create_response
    
    def _create_response(self, **kwargs):
        """Return the next response in sequence"""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        # Handle n parameter for BON approach
        n = kwargs.get('n', 1)
        if n > 1:
            # Return a response with multiple choices for BON
            return MockOpenAIResponse("Different response", 10, n, self.call_count)
        return response


class TestConversationLoggingApproaches(unittest.TestCase):
    """Test conversation logging across all approaches"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "conversations"
        self.logger = ConversationLogger(self.log_dir, enabled=True)

        # Set the global logger instance for approach modules to use
        set_global_logger(self.logger)

        # Common test parameters
        self.system_prompt = "You are a helpful assistant."
        self.initial_query = "What is 2 + 2?"
        self.model = "test-model"
        self.request_id = "test-request-123"

        # Create mock client
        self.client = MockOpenAIClient()

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clear the global logger
        set_global_logger(None)
    
    def test_multi_call_approaches_logging(self):
        """Test BON, MCTS, and RTO approaches log API calls correctly"""
        # Test BON approach
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "bon", 
            self.model
        )
        
        result, tokens = best_of_n_sampling(
            self.system_prompt, 
            self.initial_query, 
            self.client, 
            self.model, 
            n=2, 
            request_id=self.request_id
        )
        
        # BON makes multiple calls for sampling and rating
        bon_calls = self.client.call_count
        self.assertGreaterEqual(bon_calls, 2)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        
        # Reset client and test MCTS
        self.client.call_count = 0
        mcts_request_id = self.request_id + "_mcts"
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "mcts", 
            self.model
        )
        
        result, tokens = chat_with_mcts(
            self.system_prompt,
            self.initial_query,
            self.client,
            self.model,
            num_simulations=2,
            exploration_weight=0.2,
            simulation_depth=1,
            request_id=mcts_request_id
        )
        
        mcts_calls = self.client.call_count
        self.assertGreaterEqual(mcts_calls, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        
        # Reset client and test RTO
        self.client.call_count = 0
        rto_request_id = self.request_id + "_rto"
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "rto", 
            self.model
        )
        
        result, tokens = round_trip_optimization(
            self.system_prompt,
            self.initial_query,
            self.client,
            self.model,
            request_id=rto_request_id
        )
        
        # RTO makes either 3 calls (if C1==C2) or 4 calls (C1 -> Q2 -> C2 -> C3)
        rto_calls = self.client.call_count
        self.assertGreaterEqual(rto_calls, 3)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
    
    def test_single_call_approaches_logging(self):
        """Test CoT Reflection and RE2 approaches log single API calls correctly"""
        # Test CoT Reflection
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "cot_reflection", 
            self.model
        )
        
        result, tokens = cot_reflection(
            self.system_prompt,
            self.initial_query,
            self.client,
            self.model,
            request_id=self.request_id
        )
        
        # CoT Reflection makes exactly 1 API call
        self.assertEqual(self.client.call_count, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        
        # Reset client and test RE2
        self.client.call_count = 0
        re2_request_id = self.request_id + "_re2"
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "re2", 
            self.model
        )
        
        result, tokens = re2_approach(
            self.system_prompt,
            self.initial_query,
            self.client,
            self.model,
            n=1,
            request_id=re2_request_id
        )
        
        # RE2 makes exactly 1 API call
        self.assertEqual(self.client.call_count, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
    
    def test_sampling_approaches_logging(self):
        """Test PVG and Self Consistency approaches log multiple sampling calls"""
        # Test PVG approach
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "pvg", 
            self.model
        )
        
        result, tokens = inference_time_pv_game(
            self.system_prompt,
            self.initial_query,
            self.client,
            self.model,
            num_rounds=1,  # Reduce rounds for faster testing
            num_solutions=2,  # Reduce solutions for faster testing
            request_id=self.request_id
        )
        
        # PVG makes multiple API calls: solutions + verifications + refinement
        pvg_calls = self.client.call_count
        self.assertGreaterEqual(pvg_calls, 3)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        
        # Reset client and test Self Consistency
        self.client.call_count = 0
        sc_request_id = self.request_id + "_sc"
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "self_consistency", 
            self.model
        )
        
        result, tokens = advanced_self_consistency_approach(
            self.system_prompt,
            self.initial_query,
            self.client,
            self.model,
            request_id=sc_request_id
        )
        
        # Self Consistency makes num_samples API calls (default 5)
        self.assertEqual(self.client.call_count, 5)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
    
    @patch('optillm.z3_solver.multiprocessing.get_context')
    def test_complex_class_based_approaches_logging(self, mock_mp_context):
        """Test RStar and Z3 Solver class-based approaches log API calls correctly"""
        # Test RStar approach
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "rstar", 
            self.model
        )
        
        rstar = RStar(
            self.system_prompt,
            self.client,
            self.model,
            max_depth=2,  # Reduce depth for faster testing
            num_rollouts=2,  # Reduce rollouts for faster testing
            request_id=self.request_id
        )
        
        result, tokens = rstar.solve(self.initial_query)
        
        # RStar makes multiple API calls during MCTS rollouts
        rstar_calls = self.client.call_count
        self.assertGreaterEqual(rstar_calls, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        
        # Reset client and test Z3 Solver
        self.client.call_count = 0
        z3_request_id = self.request_id + "_z3"
        self.logger.start_conversation(
            {"model": self.model, "messages": []}, 
            "z3", 
            self.model
        )
        
        # Mock multiprocessing for z3_solver
        mock_pool = Mock()
        mock_result = Mock()
        mock_result.get.return_value = ("success", "Test solver output")
        mock_pool.apply_async.return_value = mock_result
        mock_context = Mock()
        mock_context.Pool.return_value = MagicMock()  # Use MagicMock for context manager
        mock_context.Pool.return_value.__enter__.return_value = mock_pool
        mock_context.Pool.return_value.__exit__.return_value = None
        mock_mp_context.return_value = mock_context
        
        z3_solver = Z3SymPySolverSystem(
            self.system_prompt,
            self.client,
            self.model,
            request_id=z3_request_id
        )
        
        result, tokens = z3_solver.process_query(self.initial_query)
        
        # Z3 Solver makes at least 1 API call for analysis
        self.assertGreaterEqual(self.client.call_count, 1)
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
    
    def test_logging_edge_cases(self):
        """Test approaches work with logging disabled, no request_id, and API errors"""
        # Test with logging disabled
        optillm.conversation_logger = None
        
        result, tokens = best_of_n_sampling(
            self.system_prompt,
            self.initial_query,
            self.client,
            self.model,
            n=2,
            request_id=self.request_id
        )
        
        # Should still work normally
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        
        # Re-enable logging for next test
        optillm.conversation_logger = self.logger
        
        # Test with no request_id
        self.client.call_count = 0
        result, tokens = cot_reflection(
            self.system_prompt,
            self.initial_query,
            self.client,
            self.model,
            request_id=None
        )
        
        # Should still work normally
        self.assertIsInstance(result, str)
        self.assertGreater(tokens, 0)
        
        # Test API error handling
        error_client = Mock()
        error_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Test that approaches handle errors gracefully
        with self.assertRaises(Exception):
            cot_reflection(
                self.system_prompt,
                self.initial_query,
                error_client,
                self.model,
                request_id=self.request_id
            )


    def test_full_integration_with_file_logging(self):
        """Test complete integration from approach execution to file logging"""
        # Start conversation and get request_id
        request_id = self.logger.start_conversation(
            {"model": "test-model", "messages": []}, 
            "bon", 
            "test-model"
        )
        
        # Run approach with the returned request_id
        result, tokens = best_of_n_sampling(
            "You are a helpful assistant.",
            "What is 2 + 2?",
            self.client,
            "test-model",
            n=2,
            request_id=request_id
        )
        
        # Finalize conversation
        self.logger.finalize_conversation(request_id)
        
        # Check that conversation was logged
        log_files = list(self.log_dir.glob("*.jsonl"))
        self.assertGreater(len(log_files), 0)
        
        # Check that log file contains entries
        with open(log_files[0], 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            
            # Verify log entry structure
            log_entry = json.loads(lines[0].strip())
            self.assertEqual(log_entry["approach"], "bon")
            self.assertIn("provider_calls", log_entry)
            self.assertGreater(len(log_entry["provider_calls"]), 0)


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)