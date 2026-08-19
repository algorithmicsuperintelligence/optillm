"""
Integration tests for MiniMax provider support in optillm.

These tests verify that the MiniMax provider works end-to-end with the actual
MiniMax API. They require a valid MINIMAX_API_KEY environment variable.

Run with: MINIMAX_API_KEY=your-key pytest tests/test_minimax_integration.py -v
"""

import unittest
import os
import sys

# Add parent directory to path to import optillm modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip all tests if MINIMAX_API_KEY is not set
MINIMAX_API_KEY = os.environ.get('MINIMAX_API_KEY', '')
SKIP_REASON = "MINIMAX_API_KEY environment variable not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the actual MiniMax API."""

    def setUp(self):
        """Set up OpenAI client pointing to MiniMax API."""
        from openai import OpenAI
        self.client = OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1"
        )

    def test_basic_completion(self):
        """Test basic chat completion with MiniMax API."""
        response = self.client.chat.completions.create(
            model="MiniMax-M3",
            messages=[
                {"role": "user", "content": "Say hello in one word."}
            ],
            max_tokens=10,
            temperature=0.7
        )

        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_temperature_boundary(self):
        """Test that MiniMax API accepts temperature at boundary value 0.01."""
        response = self.client.chat.completions.create(
            model="MiniMax-M3",
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=10,
            temperature=0.01
        )

        assert hasattr(response, 'choices')
        assert len(response.choices) > 0

    def test_streaming_completion(self):
        """Test streaming chat completion with MiniMax API."""
        stream = self.client.chat.completions.create(
            model="MiniMax-M3",
            messages=[
                {"role": "user", "content": "Count from 1 to 3."}
            ],
            max_tokens=30,
            temperature=0.5,
            stream=True
        )

        chunks = list(stream)
        assert len(chunks) > 0
        content_chunks = [
            chunk.choices[0].delta.content
            for chunk in chunks
            if chunk.choices[0].delta.content
        ]
        assert len(content_chunks) > 0


if __name__ == '__main__':
    unittest.main()
