#!/usr/bin/env python3
"""
Quick CI test to verify basic functionality
"""

import time
import sys
import os

start_time = time.time()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import key modules to ensure they load
try:
    from optillm import parse_combined_approach
    print("✅ Core optillm module imported successfully")
except Exception as e:
    print(f"❌ Failed to import core modules: {e}")
    sys.exit(1)

# Test importing approach modules
try:
    print("✅ Approach modules imported successfully")
except Exception as e:
    print(f"❌ Failed to import approach modules: {e}")

# Test plugin existence
try:
    print("✅ Basic plugin modules exist and can be imported")
except Exception as e:
    print(f"❌ Basic plugin import test failed: {e}")

# Test plugin subdirectory imports (critical for issue #220)
try:
    from optillm.plugins.deepthink import SelfDiscover, UncertaintyRoutedCoT
    from optillm.plugins.deep_research import DeepResearcher
    from optillm.plugins.longcepo import run_longcepo
    from optillm.plugins.spl import run_spl
    from optillm.plugins.proxy import client, config, approach_handler
    print("✅ Plugin submodule imports working - no relative import errors")
except ImportError as e:
    if "attempted relative import" in str(e):
        print(f"❌ Critical: Relative import error detected: {e}")
        sys.exit(1)
    else:
        print(f"❌ Plugin submodule import error: {e}")
except Exception as e:
    print(f"❌ Plugin submodule import error: {e}")

# Test approach parsing
try:
    # Define known approaches for testing
    known_approaches = ["moa", "bon", "mcts", "cot_reflection"]
    plugin_approaches_test = {"memory": True, "readurls": True}
    
    test_cases = [
        ("moa-gpt-4", "SINGLE", ["moa"], "gpt-4"),
        ("bon|moa|mcts-gpt-4", "OR", ["bon", "moa", "mcts"], "gpt-4"),
        ("memory&moa-gpt-4", "AND", ["memory", "moa"], "gpt-4"),
    ]
    
    for combined, expected_op, expected_approaches, expected_model in test_cases:
        operation, approaches, model = parse_combined_approach(combined, known_approaches, plugin_approaches_test)
        assert operation == expected_op, f"Expected operation {expected_op}, got {operation}"
        assert approaches == expected_approaches, f"Expected {expected_approaches}, got {approaches}"
        assert model == expected_model, f"Expected {expected_model}, got {model}"
    
    print("✅ Approach parsing tests passed")
except Exception as e:
    print(f"❌ Approach parsing test failed: {e}")

print("\n✅ All CI quick tests completed!")
print(f"Total test time: {time.time() - start_time:.2f}s")