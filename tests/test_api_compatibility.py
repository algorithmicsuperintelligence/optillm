#!/usr/bin/env python3
"""
Test API compatibility with OpenAI format
"""

import pytest
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test utilities
from test_utils import setup_test_env, get_test_client, TEST_MODEL


@pytest.fixture
def client():
    """Create OpenAI client for optillm proxy with local inference"""
    setup_test_env()
    return get_test_client()


def test_basic_completion(client):
    """Test basic chat completion"""
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=10
    )
    
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], 'message')
    assert hasattr(response.choices[0].message, 'content')


def test_n_parameter(client):
    """Test n parameter for multiple completions"""
    n = 3
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "user", "content": "Write a one-line joke"}
        ],
        n=n,
        temperature=0.8,
        max_tokens=50
    )
    
    assert len(response.choices) == n
    # Check all responses are different (with high temperature)
    contents = [choice.message.content for choice in response.choices]
    assert len(set(contents)) > 1  # At least some different responses


def test_approach_prefix(client):
    """Test approach prefix in model name"""
    response = client.chat.completions.create(
        model=f"moa-{TEST_MODEL}",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        max_tokens=10
    )
    
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0


def test_extra_body_approach(client):
    """Test approach specification via extra_body"""
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        extra_body={"optillm_approach": "re2"},  # Use re2 instead of bon (simpler, avoids role ordering issues with some models)
        max_tokens=50
    )

    assert hasattr(response, 'choices')
    assert len(response.choices) > 0


def test_streaming(client):
    """Test streaming response"""
    stream = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "user", "content": "Count from 1 to 5"}
        ],
        stream=True,
        max_tokens=50
    )
    
    chunks = list(stream)
    assert len(chunks) > 0
    # First chunk should have role
    assert chunks[0].choices[0].delta.role == "assistant"
    # Later chunks should have content
    content_chunks = [chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content]
    assert len(content_chunks) > 0


def test_reasoning_tokens_in_response(client):
    """Test that reasoning tokens are included in API responses"""
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "system", "content": "Think step by step and show your reasoning."},
            {"role": "user", "content": "What is 15 × 23? Please think through this step by step."}
        ],
        max_tokens=100
    )
    
    # Check basic response structure
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0
    assert hasattr(response, 'usage')
    
    # Check that completion_tokens_details exists and has reasoning_tokens
    assert hasattr(response.usage, 'completion_tokens_details')
    assert hasattr(response.usage.completion_tokens_details, 'reasoning_tokens')
    
    # reasoning_tokens should be an integer >= 0
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    assert isinstance(reasoning_tokens, int)
    assert reasoning_tokens >= 0


def test_reasoning_tokens_with_thinking_prompt(client):
    """Test reasoning tokens with a prompt designed to trigger thinking"""
    response = client.chat.completions.create(
        model=TEST_MODEL, 
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use <think> tags to show your reasoning process."},
            {"role": "user", "content": "I have 12 apples. I eat 3, give away 4, and buy 7 more. How many apples do I have now?"}
        ],
        max_tokens=150
    )
    
    # Basic checks
    assert hasattr(response, 'usage')
    assert hasattr(response.usage, 'completion_tokens_details')
    assert hasattr(response.usage.completion_tokens_details, 'reasoning_tokens')
    
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    assert isinstance(reasoning_tokens, int)
    assert reasoning_tokens >= 0
    
    # If the model used thinking tags, reasoning_tokens should be > 0
    # (This depends on the model's response, so we just check the structure)
    

def test_reasoning_tokens_with_multiple_responses(client):
    """Test reasoning tokens with n > 1"""
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "user", "content": "Think about this: What's 2+2?"}
        ],
        n=2,
        max_tokens=50
    )
    
    # Should have 2 choices
    assert len(response.choices) == 2
    
    # Should have reasoning token information
    assert hasattr(response.usage, 'completion_tokens_details')
    assert hasattr(response.usage.completion_tokens_details, 'reasoning_tokens')
    
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    assert isinstance(reasoning_tokens, int)
    assert reasoning_tokens >= 0


def test_reasoning_tokens_backward_compatibility(client):
    """Test that responses without thinking still work normally"""
    response = client.chat.completions.create(
        model=TEST_MODEL,
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=10
    )
    
    # Should still have reasoning token structure, but with 0 tokens
    assert hasattr(response.usage, 'completion_tokens_details')
    assert hasattr(response.usage.completion_tokens_details, 'reasoning_tokens')
    
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    assert isinstance(reasoning_tokens, int)
    assert reasoning_tokens >= 0  # Usually 0 for simple responses


if __name__ == "__main__":
    # Run basic tests if pytest not available
    setup_test_env()
    client = get_test_client()
    
    print("Running API compatibility tests...")
    
    tests = [
        ("Basic completion", test_basic_completion),
        ("N parameter", test_n_parameter),
        ("Approach prefix", test_approach_prefix),
        ("Extra body approach", test_extra_body_approach),
        ("Streaming", test_streaming),
        ("Reasoning tokens in response", test_reasoning_tokens_in_response),
        ("Reasoning tokens with thinking prompt", test_reasoning_tokens_with_thinking_prompt),
        ("Reasoning tokens with multiple responses", test_reasoning_tokens_with_multiple_responses),
        ("Reasoning tokens backward compatibility", test_reasoning_tokens_backward_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"Running {test_name}...", end=' ')
            test_func(client)
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print("\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed.")