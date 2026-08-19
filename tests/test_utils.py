"""
Test utilities for OptILLM tests
Provides common functions and constants for consistent testing
"""

import os
import sys
import time
import subprocess
import platform
from typing import Optional
from openai import OpenAI

# Standard test model for all tests - small and fast (~250M params)
TEST_MODEL = "codelion/dhara-250m"
# MLX build is only used on Apple Silicon; no MLX variant of dhara exists yet,
# so MLX-specific tests continue to use the Qwen build (skipped on non-macOS CI).
TEST_MODEL_MLX = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16"

def setup_test_env():
    """Set up test environment with local inference"""
    os.environ["OPTILLM_API_KEY"] = "optillm"
    return TEST_MODEL

def get_test_client(base_url: str = "http://localhost:8000/v1") -> OpenAI:
    """Get OpenAI client configured for local optillm"""
    return OpenAI(api_key="optillm", base_url=base_url)

def is_mlx_available():
    """Check if MLX is available (macOS only)"""
    if platform.system() != "Darwin":
        return False
    try:
        from optillm.inference import MLX_AVAILABLE
        return MLX_AVAILABLE
    except ImportError:
        return False

def start_test_server(model: str = TEST_MODEL, port: int = 8000) -> subprocess.Popen:
    """
    Start optillm server for testing
    Returns the process handle
    """
    # Set environment for local inference
    env = os.environ.copy()
    env["OPTILLM_API_KEY"] = "optillm"
    # Enable MPS fallback to CPU for unsupported operations (fixes macOS compatibility)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Get the project root directory (parent of tests directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Start server from project root where optillm.py is located
    proc = subprocess.Popen([
        sys.executable, "optillm.py",
        "--model", model,
        "--port", str(port)
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project_root)

    # Wait for server to start
    time.sleep(5)

    return proc

def stop_test_server(proc: subprocess.Popen):
    """Stop the test server"""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

def get_simple_test_messages():
    """Get simple test messages for basic validation"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one word."}
    ]

def get_math_test_messages():
    """Get math test messages for reasoning validation"""
    return [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
    ]

def get_thinking_test_messages():
    """Get test messages that should generate thinking tokens"""
    return [
        {"role": "system", "content": "Think step by step and use <think></think> tags."},
        {"role": "user", "content": "What is 3 * 4? Show your thinking."}
    ]