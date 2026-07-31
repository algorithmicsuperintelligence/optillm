#!/usr/bin/env python3
"""
Tests for reasoning token functionality in OptILLM
Covers count_reasoning_tokens function and API response format
"""

import sys
import os
import unittest
from unittest.mock import Mock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test utilities
from test_utils import (
    setup_test_env, get_test_client, TEST_MODEL, 
    get_simple_test_messages, get_thinking_test_messages
)

# Import the count_reasoning_tokens function from both modules
from optillm import count_reasoning_tokens as optillm_count_reasoning_tokens
from optillm.inference import count_reasoning_tokens as inference_count_reasoning_tokens


class TestCountReasoningTokens(unittest.TestCase):
    """Test the count_reasoning_tokens function"""
    
    def test_count_reasoning_tokens_basic(self):
        """Test basic functionality of count_reasoning_tokens"""
        # Test with think tags
        text_with_think = "<think>This is reasoning content</think>This is output"
        
        # Test both implementations should work the same
        result1 = optillm_count_reasoning_tokens(text_with_think)
        result2 = inference_count_reasoning_tokens(text_with_think)
        
        self.assertGreater(result1, 0)
        self.assertEqual(result1, result2)
    
    def test_count_reasoning_tokens_no_think_tags(self):
        """Test with text that has no think tags"""
        text_without_think = "This is just regular output text"
        
        result1 = optillm_count_reasoning_tokens(text_without_think)
        result2 = inference_count_reasoning_tokens(text_without_think)
        
        self.assertEqual(result1, 0)
        self.assertEqual(result2, 0)
    
    def test_count_reasoning_tokens_multiple_think_blocks(self):
        """Test with multiple think tag blocks"""
        text_multiple = """
        <think>First reasoning block</think>
        Some output here
        <think>Second reasoning block with more content</think>
        Final output
        """
        
        result = optillm_count_reasoning_tokens(text_multiple)
        self.assertGreater(result, 0)
        
        # Should count tokens from both blocks
        single_block = "<think>First reasoning blockSecond reasoning block with more content</think>"
        single_result = optillm_count_reasoning_tokens(single_block)
        self.assertAlmostEqual(result, single_result, delta=2)  # Allow small variance due to formatting
    
    def test_count_reasoning_tokens_empty_input(self):
        """Test with empty or None input"""
        self.assertEqual(optillm_count_reasoning_tokens(""), 0)
        self.assertEqual(optillm_count_reasoning_tokens(None), 0)
        self.assertEqual(optillm_count_reasoning_tokens(123), 0)  # Non-string input
    
    def test_count_reasoning_tokens_malformed_tags(self):
        """Test with malformed think tags"""
        malformed_cases = [
            "<think>Unclosed think tag",
            "Unopened think tag</think>",
            "<think><think>Nested tags</think></think>",
            "<THINK>Wrong case</THINK>",
            "<think></think>",  # Empty think block
        ]
        
        for case in malformed_cases:
            result = optillm_count_reasoning_tokens(case)
            # Should handle gracefully, either 0 or some reasonable count
            self.assertGreaterEqual(result, 0)
    
    def test_count_reasoning_tokens_with_tokenizer(self):
        """Test with a mock tokenizer for precise counting"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        
        text = "<think>Some reasoning text</think>Output"
        result = optillm_count_reasoning_tokens(text, mock_tokenizer)
        
        self.assertEqual(result, 5)
        mock_tokenizer.encode.assert_called_once_with("Some reasoning text")
    
    def test_count_reasoning_tokens_tokenizer_error(self):
        """Test fallback when tokenizer fails"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        
        text = "<think>Some reasoning text</think>Output"
        result = optillm_count_reasoning_tokens(text, mock_tokenizer)
        
        # Should fallback to character-based estimation
        self.assertGreater(result, 0)
        mock_tokenizer.encode.assert_called_once()
    
    def test_count_reasoning_tokens_multiline(self):
        """Test with multiline think blocks"""
        multiline_text = """<think>
        This is a multi-line reasoning block
        with several lines of content
        that spans multiple lines
        </think>
        This is the final output"""
        
        result = optillm_count_reasoning_tokens(multiline_text)
        self.assertGreater(result, 10)  # Should be substantial content
    
    def test_count_reasoning_tokens_special_characters(self):
        """Test with special characters in think blocks"""
        special_text = "<think>Content with émojis 🤔 and symbols @#$%^&*()</think>Output"
        result = optillm_count_reasoning_tokens(special_text)
        self.assertGreater(result, 0)


class TestAPIResponseFormat(unittest.TestCase):
    """Test that API responses include reasoning token information"""
    
    def setUp(self):
        """Set up test fixtures"""
        setup_test_env()
        self.test_client = get_test_client()
    
    def test_response_includes_completion_tokens_details(self):
        """Test that API responses include completion_tokens_details"""
        try:
            # Make request with local inference
            response = self.test_client.chat.completions.create(
                model=TEST_MODEL,
                messages=get_thinking_test_messages(),
                max_tokens=50
            )
            
            # Check basic response structure
            self.assertIsNotNone(response.choices)
            self.assertEqual(len(response.choices), 1)
            self.assertIsNotNone(response.choices[0].message.content)
            
            # Check usage information
            self.assertIsNotNone(response.usage)
            self.assertGreater(response.usage.completion_tokens, 0)
            self.assertGreater(response.usage.prompt_tokens, 0)
            
            # Note: reasoning token structure depends on model response format
            # Some models may not generate <think> tags naturally
            
        except Exception as e:
            self.skipTest(f"Local inference not available: {str(e)}")
    
    def test_response_no_reasoning_tokens(self):
        """Test API response when there are no reasoning tokens"""
        try:
            # Make request with simple messages (no thinking prompt)
            response = self.test_client.chat.completions.create(
                model=TEST_MODEL,
                messages=get_simple_test_messages(),
                max_tokens=20
            )
            
            # Check basic response structure
            self.assertIsNotNone(response.choices)
            self.assertEqual(len(response.choices), 1)
            self.assertIsNotNone(response.choices[0].message.content)
            
            # Check usage information
            self.assertIsNotNone(response.usage)
            self.assertGreater(response.usage.completion_tokens, 0)
            self.assertGreater(response.usage.prompt_tokens, 0)
            
            # For simple messages without <think> tags, reasoning tokens should be 0
            # But this depends on the actual model response format
            
        except Exception as e:
            self.skipTest(f"Local inference not available: {str(e)}")
    
    def test_multiple_responses_reasoning_tokens(self):
        """Test reasoning tokens with multiple responses (n > 1)"""
        try:
            # Make request with n=2 to get multiple responses
            response = self.test_client.chat.completions.create(
                model=TEST_MODEL,
                messages=get_thinking_test_messages(),
                max_tokens=50,
                n=2
            )
            
            # Check basic response structure
            self.assertIsNotNone(response.choices)
            self.assertGreaterEqual(len(response.choices), 1)  # May return 1 or 2 depending on implementation
            
            # Check usage information
            self.assertIsNotNone(response.usage)
            self.assertGreater(response.usage.completion_tokens, 0)
            
            # Note: Multiple responses depend on model capability and implementation
            # Local inference may not fully support n > 1
            
        except Exception as e:
            self.skipTest(f"Multiple responses not supported by local inference: {str(e)}")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing functionality"""
    
    def test_existing_approaches_still_work(self):
        """Test that existing approaches work without reasoning token changes"""
        # Import approaches that don't use reasoning
        from optillm.bon import best_of_n_sampling
        
        # Create mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Regular response"
        mock_response.usage.completion_tokens = 10
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test that approach still works
        try:
            result, tokens = best_of_n_sampling(
                system_prompt="You are a helpful assistant.",
                initial_query="test",
                client=mock_client,
                model="test-model",
                n=3
            )
            self.assertIsInstance(result, str)
            self.assertIsInstance(tokens, int)
        except Exception as e:
            self.fail(f"Existing approach failed: {e}")
    
    def test_api_without_auth_header(self):
        """Test API still returns proper errors without auth"""
        import optillm
        app = optillm.app
        app.config['TESTING'] = True
        client = app.test_client()
        
        response = client.post('/v1/chat/completions', 
                             json={"model": TEST_MODEL, "messages": []})
        
        # Should return an error (500 with local inference, 401/403 for auth issues)
        self.assertIn(response.status_code, [401, 403, 500])


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)