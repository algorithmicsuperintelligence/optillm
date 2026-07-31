#!/usr/bin/env python3
"""
Test script to verify n parameter works correctly with optillm
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test utilities
from test_utils import setup_test_env, get_test_client, TEST_MODEL

def test_n_parameter(model=TEST_MODEL, n_values=[1, 2, 3]):
    """
    Test the n parameter with different values
    """
    # Set up test environment and get client
    setup_test_env()
    client = get_test_client()
    
    test_prompt = "Write a haiku about coding"
    
    for n in n_values:
        print(f"\nTesting n={n} with model {model}")
        print("-" * 50)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a creative poet."},
                    {"role": "user", "content": test_prompt}
                ],
                n=n,
                temperature=0.8,
                max_tokens=100
            )
            
            # Check response structure
            print(f"Response type: {type(response)}")
            print(f"Number of choices: {len(response.choices)}")
            
            # Print all generated responses
            for i, choice in enumerate(response.choices):
                print(f"\nChoice {i+1}:")
                print(choice.message.content)
            
            # Verify we got the expected number of responses
            if len(response.choices) == n:
                print(f"\n✅ SUCCESS: Got {n} responses as expected")
            else:
                print(f"\n❌ FAIL: Expected {n} responses, got {len(response.choices)}")
                
        except Exception as e:
            print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")

def main():
    """
    Main test function
    """
    print("Testing n parameter support in optillm")
    print("=" * 50)
    
    # Set up test environment
    setup_test_env()
    
    # Use the standard test model
    model = TEST_MODEL
    print(f"\n\nTesting model: {model}")
    print("=" * 50)
    
    try:
        test_n_parameter(model)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("Make sure optillm server is running with local inference enabled")
        return 1
    
    return 0

if __name__ == "__main__":
    main()