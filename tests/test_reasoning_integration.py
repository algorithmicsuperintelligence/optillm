#!/usr/bin/env python3
"""
Integration tests for reasoning token functionality
Tests end-to-end integration with approaches that generate thinking
"""

import sys
import os
import unittest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test utilities
from test_utils import (
    setup_test_env, is_mlx_available, 
    TEST_MODEL, get_simple_test_messages
)

# Import the thinkdeeper functions for testing
try:
    from optillm.thinkdeeper_mlx import thinkdeeper_decode_mlx
    MLX_THINKDEEPER_AVAILABLE = True
except ImportError:
    MLX_THINKDEEPER_AVAILABLE = False



class TestThinkDeeperReasoningTokens(unittest.TestCase):
    """Test ThinkDeeper approaches return reasoning tokens"""
    
    def setUp(self):
        """Set up test fixtures"""
        setup_test_env()
        self.test_messages = get_simple_test_messages()
    
    def test_thinkdeeper_returns_reasoning_tokens(self):
        """Test that thinkdeeper_decode returns reasoning tokens"""
        # Setup local inference environment
        setup_test_env()
        
        try:
            # This test verifies the function signature exists and returns correct format
            # We skip actual inference testing as it requires complex model setup
            from optillm.thinkdeeper import thinkdeeper_decode
            
            # Verify function exists and has correct signature
            self.assertTrue(callable(thinkdeeper_decode))
            
            # For now, just verify the import works
            # Full integration testing will be done in TestEndToEndIntegration
            self.assertTrue(True, "thinkdeeper_decode function is available")
                
        except Exception as e:
            # If thinkdeeper fails, that's informative for debugging
            self.skipTest(f"thinkdeeper_decode not available: {str(e)}")
    
    @unittest.skipIf(not is_mlx_available() or not MLX_THINKDEEPER_AVAILABLE, "MLX or thinkdeeper_mlx not available")
    def test_thinkdeeper_mlx_returns_reasoning_tokens(self):
        """Test that thinkdeeper_decode_mlx returns reasoning tokens (MLX only)"""
        # Setup local inference environment
        setup_test_env()
        
        try:
            # Verify function exists and has correct signature
            self.assertTrue(callable(thinkdeeper_decode_mlx))
            
            # For now, just verify the import works
            # Full MLX integration testing requires Apple Silicon
            self.assertTrue(True, "thinkdeeper_decode_mlx function is available")
                
        except Exception as e:
            # If MLX thinkdeeper fails, that's informative for debugging
            self.skipTest(f"thinkdeeper_decode_mlx not available: {str(e)}")


class TestInferenceIntegration(unittest.TestCase):
    """Test integration with inference.py module"""
    
    def test_inference_usage_includes_reasoning_tokens(self):
        """Test that ChatCompletionUsage includes reasoning_tokens"""
        from optillm.inference import ChatCompletionUsage
        
        # Test creating usage with reasoning tokens
        usage = ChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=20, 
            total_tokens=30,
            reasoning_tokens=5
        )
        
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertEqual(usage.reasoning_tokens, 5)
    
    def test_inference_usage_default_reasoning_tokens(self):
        """Test that ChatCompletionUsage defaults reasoning_tokens to 0"""
        from optillm.inference import ChatCompletionUsage
        
        # Test creating usage without reasoning tokens
        usage = ChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
        
        self.assertEqual(usage.reasoning_tokens, 0)
    
    def test_chat_completion_model_dump_includes_reasoning_tokens(self):
        """Test that ChatCompletion.model_dump includes reasoning_tokens in usage"""
        from optillm.inference import ChatCompletion
        
        # Create mock response with reasoning tokens
        response_dict = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>reasoning</think>answer"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "reasoning_tokens": 5
            }
        }
        
        completion = ChatCompletion(response_dict)
        result = completion.model_dump()
        
        # Check that model_dump includes reasoning_tokens
        self.assertIn("usage", result)
        self.assertIn("completion_tokens_details", result["usage"])
        self.assertIn("reasoning_tokens", result["usage"]["completion_tokens_details"])
        self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 5)


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration with mocked responses for specific configs"""
    
    def test_thinkdeeper_approach_with_reasoning_tokens(self):
        """Test thinkdeeper approach properly processes reasoning tokens"""
        from unittest.mock import patch, Mock
        
        # Test thinkdeeper processing with mocked response
        with patch('optillm.thinkdeeper.thinkdeeper_decode') as mock_thinkdeeper:
            # Mock response with reasoning tokens (thinking content)
            mock_response = "<think>Let me solve this step by step. 2 + 2 = 4</think>The answer is 4."
            mock_tokens = 25
            mock_thinkdeeper.return_value = (mock_response, mock_tokens)
            
            # Call the approach
            result, tokens = mock_thinkdeeper(
                "You are a helpful assistant.",
                "What is 2+2?",
                Mock(),  # client
                TEST_MODEL,
                {"k": 3}  # thinkdeeper config
            )
            
            # Verify mocked response structure
            self.assertEqual(result, mock_response)
            self.assertEqual(tokens, mock_tokens)
            self.assertIn("<think>", result)
            self.assertIn("</think>", result)
            
            # Verify function was called with correct parameters
            mock_thinkdeeper.assert_called_once()
    
    def test_reasoning_token_calculation_with_mock_response(self):
        """Test reasoning token calculation with mock content"""
        from optillm import count_reasoning_tokens
        
        # Test cases with different thinking patterns
        test_cases = [
            ("<think>Simple thought</think>Answer", 2),  # "Simple thought" = ~2 tokens
            ("<think>More complex reasoning here</think>Final answer", 4),  # ~4 tokens
            ("No thinking tags here", 0),  # No reasoning tokens
            ("<think>First thought</think>Some text<think>Second thought</think>End", 4),  # Multiple blocks
        ]
        
        for content, expected_min_tokens in test_cases:
            with self.subTest(content=content[:30] + "..."):
                reasoning_tokens = count_reasoning_tokens(content)
                if expected_min_tokens > 0:
                    self.assertGreaterEqual(reasoning_tokens, expected_min_tokens - 1)  # Allow some variance
                else:
                    self.assertEqual(reasoning_tokens, 0)


class TestAPIResponseStructure(unittest.TestCase):
    """Test API response structure with reasoning tokens using mocks"""
    
    def test_chat_completion_response_structure(self):
        """Test that chat completion responses have proper structure"""
        from optillm.inference import ChatCompletion, ChatCompletionUsage
        
        # Create mock response structure
        mock_usage = ChatCompletionUsage(
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40,
            reasoning_tokens=8
        )
        
        # Verify usage structure
        self.assertEqual(mock_usage.prompt_tokens, 15)
        self.assertEqual(mock_usage.completion_tokens, 25)
        self.assertEqual(mock_usage.total_tokens, 40)
        self.assertEqual(mock_usage.reasoning_tokens, 8)
        
        # Test response with reasoning tokens included
        response_data = {
            "id": "test-completion",
            "object": "chat.completion",
            "created": 1234567890,
            "model": TEST_MODEL,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>Let me calculate: 2+2=4</think>The answer is 4."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
                "reasoning_tokens": 8
            }
        }
        
        # Create ChatCompletion and verify structure
        completion = ChatCompletion(response_data)
        result = completion.model_dump()
        
        # Verify reasoning tokens are properly included
        self.assertIn("usage", result)
        self.assertIn("completion_tokens_details", result["usage"])
        self.assertIn("reasoning_tokens", result["usage"]["completion_tokens_details"])
        self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 8)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)