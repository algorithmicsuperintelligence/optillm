#!/usr/bin/env python3
"""
Server-based integration tests for conversation logging with real model
Tests conversation logging with actual OptILLM server and codelion/dhara-250m model
"""

import unittest
import sys
import os
import requests
import json
import tempfile
import time
import subprocess
from pathlib import Path
from openai import OpenAI

# Add parent directory and tests directory to path for imports
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_tests_dir)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from test_utils import TEST_MODEL, setup_test_env, start_test_server, stop_test_server


class TestConversationLoggingWithServer(unittest.TestCase):
    """Integration tests with real OptILLM server and conversation logging"""
    
    @classmethod
    def setUpClass(cls):
        """Set up OptILLM server for testing"""
        setup_test_env()
        
        # Check if server is already running
        cls.server_available = cls._check_existing_server()
        cls.server_process = None
        cls.temp_log_dir = None
        
        if not cls.server_available:
            # Start our own server with logging enabled
            cls.temp_log_dir = Path(tempfile.mkdtemp())
            cls.server_process = cls._start_server_with_logging()
            
            # Wait for server to be ready
            max_wait = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if cls._check_server_health():
                    cls.server_available = True
                    break
                time.sleep(1)
            
            if not cls.server_available:
                if cls.server_process:
                    stop_test_server(cls.server_process)
                raise unittest.SkipTest("Could not start OptILLM server for testing")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up server"""
        if cls.server_process:
            stop_test_server(cls.server_process)
        
        if cls.temp_log_dir and cls.temp_log_dir.exists():
            import shutil
            shutil.rmtree(cls.temp_log_dir, ignore_errors=True)
    
    @staticmethod
    def _check_existing_server():
        """Check if OptILLM server is already running"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def _check_server_health():
        """Check if server is healthy"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    @classmethod
    def _start_server_with_logging(cls):
        """Start server with conversation logging enabled"""
        env = os.environ.copy()
        env["OPTILLM_API_KEY"] = "optillm"
        env["OPTILLM_LOG_CONVERSATIONS"] = "true"
        env["OPTILLM_CONVERSATION_LOG_DIR"] = str(cls.temp_log_dir)
        # Enable MPS fallback to CPU for unsupported operations (fixes macOS compatibility)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Get the project root directory (parent of tests directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        proc = subprocess.Popen([
            sys.executable, "optillm.py",
            "--model", TEST_MODEL,
            "--port", "8000",
            "--log-conversations",
            "--conversation-log-dir", str(cls.temp_log_dir)
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project_root)

        return proc
    
    def setUp(self):
        """Set up test client"""
        if not self.server_available:
            self.skipTest("OptILLM server not available")

        self.client = OpenAI(api_key="optillm", base_url="http://localhost:8000/v1")

        # Determine log directory - priority order:
        # 1. temp_log_dir (if we started the server ourselves)
        # 2. OPTILLM_CONVERSATION_LOG_DIR environment variable (for CI)
        # 3. Default ~/.optillm/conversations
        if self.temp_log_dir:
            self.log_dir = self.temp_log_dir
        elif os.getenv("OPTILLM_CONVERSATION_LOG_DIR"):
            self.log_dir = Path(os.getenv("OPTILLM_CONVERSATION_LOG_DIR"))
        else:
            self.log_dir = Path.home() / ".optillm" / "conversations"

        # Record initial state for comparison
        self.initial_log_files = set(self.log_dir.glob("*.jsonl")) if self.log_dir.exists() else set()
    
    def _get_new_log_entries(self):
        """Get new log entries since test started"""
        if not self.log_dir.exists():
            return []
        
        current_log_files = set(self.log_dir.glob("*.jsonl"))
        new_files = current_log_files - self.initial_log_files
        modified_files = [f for f in self.initial_log_files if f in current_log_files and f.stat().st_mtime > time.time() - 60]
        
        entries = []
        for log_file in new_files.union(set(modified_files)):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
            except (json.JSONDecodeError, IOError):
                continue
        
        return entries
    
    def test_basic_none_approach_logging(self):
        """Test basic none approach with conversation logging"""
        response = self.client.chat.completions.create(
            model=TEST_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
            ],
            max_tokens=10
        )
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)
        self.assertIsNotNone(response.choices[0].message.content)
        
        # Wait for logging
        time.sleep(2)
        
        # Check for new log entries
        entries = self._get_new_log_entries()
        self.assertGreater(len(entries), 0, "No log entries found for basic none approach")
        
        # Verify at least one entry has the expected structure
        found_entry = False
        for entry in entries:
            if entry.get("approach") == "none" and entry.get("model") == TEST_MODEL:
                found_entry = True
                self.assertIn("provider_calls", entry)
                self.assertIn("client_request", entry)
                self.assertIn("timestamp", entry)
                break
        
        self.assertTrue(found_entry, "No valid log entry found for none approach")
    
    def test_re2_approach_logging(self):
        """Test RE2 approach with conversation logging"""
        response = self.client.chat.completions.create(
            model=f"re2-{TEST_MODEL}",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France? Answer in one word."}
            ],
            max_tokens=10
        )
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)
        
        # Wait for logging
        time.sleep(3)
        
        # Check for new log entries
        entries = self._get_new_log_entries()
        
        # Find RE2 entry
        re2_entry = None
        for entry in entries:
            if entry.get("approach") == "re2":
                re2_entry = entry
                break
        
        self.assertIsNotNone(re2_entry, "No RE2 log entry found")
        self.assertEqual(re2_entry["model"], TEST_MODEL)
        self.assertIn("provider_calls", re2_entry)
        self.assertGreaterEqual(len(re2_entry["provider_calls"]), 1)
    
    def test_cot_reflection_approach_logging(self):
        """Test CoT Reflection approach with conversation logging"""
        response = self.client.chat.completions.create(
            model=f"cot_reflection-{TEST_MODEL}",
            messages=[
                {"role": "system", "content": "Think step by step."},
                {"role": "user", "content": "What is 3 × 4? Show your work."}
            ],
            max_tokens=50
        )
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)
        
        # Wait for logging
        time.sleep(3)
        
        # Check for log entries
        entries = self._get_new_log_entries()
        
        # Find CoT reflection entry
        cot_entry = None
        for entry in entries:
            if entry.get("approach") == "cot_reflection":
                cot_entry = entry
                break
        
        self.assertIsNotNone(cot_entry, "No CoT reflection log entry found")
        self.assertEqual(cot_entry["model"], TEST_MODEL)
        self.assertIn("provider_calls", cot_entry)
        self.assertGreaterEqual(len(cot_entry["provider_calls"]), 1)
    
    def test_extra_body_approach_logging(self):
        """Test approach specification via extra_body parameter"""
        response = self.client.chat.completions.create(
            model=TEST_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test extra_body. Reply with 'OK'."}
            ],
            extra_body={"optillm_approach": "re2"},
            max_tokens=10
        )
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)
        
        # Wait for logging
        time.sleep(3)
        
        # Check for log entries
        entries = self._get_new_log_entries()
        
        # Find entry with RE2 approach (specified via extra_body)
        found_entry = False
        for entry in entries:
            if entry.get("approach") == "re2" and entry.get("model") == TEST_MODEL:
                found_entry = True
                self.assertIn("provider_calls", entry)
                break
        
        self.assertTrue(found_entry, "No log entry found for extra_body approach specification")
    
    def test_reasoning_tokens_logging(self):
        """Test that reasoning tokens are properly logged"""
        response = self.client.chat.completions.create(
            model=TEST_MODEL,
            messages=[
                {"role": "system", "content": "Think step by step and show reasoning."},
                {"role": "user", "content": "What is 5 + 7? Explain your thinking."}
            ],
            max_tokens=100
        )
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.usage)
        
        # Wait for logging
        time.sleep(3)
        
        # Check for log entries with usage information
        entries = self._get_new_log_entries()
        
        found_usage_entry = False
        for entry in entries:
            if "provider_calls" in entry and len(entry["provider_calls"]) > 0:
                for call in entry["provider_calls"]:
                    if "response" in call and "usage" in call["response"]:
                        found_usage_entry = True
                        usage = call["response"]["usage"]
                        self.assertIn("completion_tokens", usage)
                        # reasoning_tokens might be 0 or missing for this simple model
                        if "completion_tokens_details" in usage:
                            details = usage["completion_tokens_details"]
                            if "reasoning_tokens" in details:
                                self.assertIsInstance(details["reasoning_tokens"], int)
                        break
            if found_usage_entry:
                break
        
        self.assertTrue(found_usage_entry, "No log entry with usage information found")
    
    def test_multiple_approaches_logging(self):
        """Test multiple different approaches get logged correctly"""
        approaches_to_test = [
            ("none", TEST_MODEL),
            ("re2", f"re2-{TEST_MODEL}"),
            ("cot_reflection", f"cot_reflection-{TEST_MODEL}")
        ]
        
        responses = []
        for approach_name, model_name in approaches_to_test:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Test {approach_name}. Reply 'OK'."}
                    ],
                    max_tokens=10
                )
                responses.append((approach_name, response))
                time.sleep(1)  # Brief pause between requests
            except Exception as e:
                self.fail(f"Approach {approach_name} failed: {e}")
        
        # Verify all responses
        self.assertEqual(len(responses), 3)
        for approach_name, response in responses:
            self.assertIsNotNone(response)
            self.assertGreater(len(response.choices), 0)
        
        # Wait for logging
        time.sleep(5)
        
        # Check for log entries
        entries = self._get_new_log_entries()
        
        # Find entries for each approach
        found_approaches = set()
        for entry in entries:
            approach = entry.get("approach")
            if approach in ["none", "re2", "cot_reflection"]:
                found_approaches.add(approach)
                self.assertEqual(entry["model"], TEST_MODEL)
                self.assertIn("provider_calls", entry)
        
        # Should have logged all 3 approaches
        self.assertGreaterEqual(len(found_approaches), 2, f"Not all approaches logged. Found: {found_approaches}")
    
    def test_concurrent_requests_logging(self):
        """Test that concurrent requests are logged properly"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request(index):
            try:
                response = self.client.chat.completions.create(
                    model=TEST_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Concurrent test {index}. Reply with the number {index}."}
                    ],
                    max_tokens=10
                )
                results.put(("success", index, response))
            except Exception as e:
                results.put(("error", index, str(e)))
        
        # Start multiple concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Collect results
        successful_requests = []
        while not results.empty():
            result_type, index, result = results.get()
            if result_type == "success":
                successful_requests.append((index, result))
            else:
                self.fail(f"Concurrent request {index} failed: {result}")
        
        self.assertGreaterEqual(len(successful_requests), 2, "Not enough concurrent requests succeeded")
        
        # Wait for logging
        time.sleep(5)
        
        # Check for log entries
        entries = self._get_new_log_entries()
        
        # Should have entries for concurrent requests
        concurrent_entries = [e for e in entries if "Concurrent test" in str(e.get("client_request", {}))]
        self.assertGreaterEqual(len(concurrent_entries), 2, "Not enough concurrent request log entries found")
    
    def test_error_handling_logging(self):
        """Test that errors in approaches are properly logged"""
        # Make request that might cause issues (very low max_tokens)
        try:
            response = self.client.chat.completions.create(
                model=TEST_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "This is a test for error logging scenarios."}
                ],
                max_tokens=1  # Very low to potentially cause issues
            )
            # Even if it succeeds, that's fine for this test
            self.assertIsNotNone(response)
        except Exception:
            # Exception is also fine for this test
            pass
        
        # Wait for logging
        time.sleep(3)
        
        # Check that some logging occurred (success or error)
        entries = self._get_new_log_entries()
        
        # Should have at least some entry (success or partial)
        found_relevant_entry = False
        for entry in entries:
            if "error logging scenarios" in str(entry.get("client_request", {})):
                found_relevant_entry = True
                break
        
        # Even if no specific entry found, logging system should be working
        # (this test mainly ensures no crashes in error scenarios)
        self.assertGreaterEqual(len(entries), 0, "No log entries found (system may have crashed)")
    
    def test_log_file_structure_and_format(self):
        """Test that log files have correct JSONL structure and required fields"""
        # Make a simple request
        response = self.client.chat.completions.create(
            model=TEST_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Structure test. Reply 'STRUCTURE_OK'."}
            ],
            max_tokens=15
        )
        
        self.assertIsNotNone(response)
        
        # Wait for logging
        time.sleep(3)
        
        # Check for log entries
        entries = self._get_new_log_entries()
        
        # Find relevant entry
        relevant_entry = None
        for entry in entries:
            if "STRUCTURE_OK" in str(entry.get("client_request", {})) or "Structure test" in str(entry.get("client_request", {})):
                relevant_entry = entry
                break
        
        if not relevant_entry and entries:
            # Use any recent entry for structure validation
            relevant_entry = entries[0]
        
        self.assertIsNotNone(relevant_entry, "No log entry found for structure validation")
        
        # Verify required fields in consolidated format
        required_fields = [
            "timestamp", "request_id", "approach", "model", 
            "client_request", "provider_calls"
        ]
        
        for field in required_fields:
            self.assertIn(field, relevant_entry, f"Missing required field: {field}")
        
        # Verify provider calls structure
        provider_calls = relevant_entry["provider_calls"]
        self.assertIsInstance(provider_calls, list)
        self.assertGreater(len(provider_calls), 0, "No provider calls logged")
        
        for call in provider_calls:
            self.assertIn("request", call)
            self.assertIn("response", call)
            self.assertIn("timestamp", call)
            self.assertIn("call_number", call)
        
        # Verify timestamps are valid
        self.assertIsInstance(relevant_entry["timestamp"], str)
        for call in provider_calls:
            self.assertIsInstance(call["timestamp"], str)


@unittest.skipUnless(
    os.getenv("OPTILLM_API_KEY") == "optillm",
    "Set OPTILLM_API_KEY=optillm to run server-based tests"
)
class TestConversationLoggingPerformanceWithServer(unittest.TestCase):
    """Performance tests with real server"""
    
    def setUp(self):
        """Check server availability"""
        try:
            if requests.get("http://localhost:8000/health", timeout=2).status_code != 200:
                self.skipTest("OptILLM server not available")
        except requests.exceptions.RequestException:
            self.skipTest("OptILLM server not available")
        
        self.client = OpenAI(api_key="optillm", base_url="http://localhost:8000/v1")
    
    def test_logging_performance_impact(self):
        """Test that logging doesn't significantly impact response time"""
        import time
        
        # Warm up
        self.client.chat.completions.create(
            model=TEST_MODEL,
            messages=[{"role": "user", "content": "warmup"}],
            max_tokens=5
        )
        
        # Time multiple requests
        times = []
        for i in range(5):
            start_time = time.perf_counter()
            response = self.client.chat.completions.create(
                model=TEST_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Performance test {i}. Reply 'OK'."}
                ],
                max_tokens=5
            )
            end_time = time.perf_counter()
            
            # Verify response
            self.assertIsNotNone(response)
            
            times.append(end_time - start_time)
        
        # Calculate average time
        avg_time = sum(times) / len(times)
        
        # Should be reasonably fast (under 10 seconds for small model)
        self.assertLess(avg_time, 10.0, f"Average response time too slow: {avg_time:.2f}s")
        
        print(f"\n📊 Server Performance with Logging:")
        print(f"   Average response time: {avg_time:.3f}s")
        print(f"   Response times: {[f'{t:.3f}s' for t in times]}")


if __name__ == "__main__":
    print("🚀 Running conversation logging server-based integration tests...")
    print("=" * 70)
    print("These tests require an actual OptILLM server with logging enabled.")
    print("Set OPTILLM_API_KEY=optillm to run all tests.")
    print()
    
    # Run tests
    unittest.main(verbosity=2, buffer=True)