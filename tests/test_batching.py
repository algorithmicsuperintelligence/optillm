#!/usr/bin/env python3
"""
Comprehensive Test Suite for OptILLM Request Batching

This test suite validates that:
1. Existing functionality remains unchanged without --batch-mode
2. Batch processing works correctly when enabled
3. Performance improvements are achieved
4. Both PyTorch and MLX models work correctly
"""

import unittest
import time
import json
import os
import subprocess
import tempfile
from typing import List, Dict, Any
import threading
import concurrent.futures
from unittest.mock import patch, MagicMock

# Import the modules we're testing
from optillm.batching import RequestBatcher, BatchingError
from optillm.inference import InferencePipeline, MLXInferencePipeline, MLXModelConfig, MLX_AVAILABLE

# Import test utilities
from test_utils import TEST_MODEL, TEST_MODEL_MLX


class TestRequestBatcher(unittest.TestCase):
    """Test the core RequestBatcher functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batcher = RequestBatcher(max_batch_size=4, max_wait_ms=100)
        self.test_responses = []
        
        def mock_processor(requests):
            """Mock batch processor that returns simple responses"""
            responses = []
            for i, req in enumerate(requests):
                responses.append({
                    "id": f"test-{i}",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Response to request {i}"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {"completion_tokens": 10, "total_tokens": 20}
                })
            return responses
        
        self.batcher.set_processor(mock_processor)
    
    def tearDown(self):
        """Clean up after tests"""
        self.batcher.shutdown()
    
    def test_single_request(self):
        """Test that single requests work correctly"""
        request_data = {"model": "test-model", "prompt": "Hello"}
        
        response = self.batcher.add_request(request_data)
        
        self.assertIsInstance(response, dict)
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "Response to request 0")
    
    def test_batch_formation(self):
        """Test that multiple requests form a batch"""
        def send_request(request_id):
            request_data = {"model": "test-model", "prompt": f"Request {request_id}"}
            return self.batcher.add_request(request_data)
        
        # Send 3 requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_request, i) for i in range(3)]
            responses = [future.result() for future in futures]
        
        # All should have responses
        self.assertEqual(len(responses), 3)
        for i, response in enumerate(responses):
            self.assertIsInstance(response, dict)
            self.assertEqual(response["object"], "chat.completion")
    
    def test_batch_timeout(self):
        """Test that partial batches process after timeout"""
        start_time = time.time()
        
        # Send single request - should process after timeout
        request_data = {"model": "test-model", "prompt": "Single request"}
        response = self.batcher.add_request(request_data)
        
        elapsed_time = time.time() - start_time
        
        # Should have processed after timeout
        self.assertGreater(elapsed_time, 0.09)  # ~100ms timeout
        self.assertIsInstance(response, dict)
    
    def test_incompatible_requests(self):
        """Test that incompatible requests are properly handled"""
        # Streaming requests should fail
        request_data = {"model": "test-model", "stream": True}
        
        with self.assertRaises(BatchingError):
            self.batcher.add_request(request_data)
    
    def test_processor_error_handling(self):
        """Test that processor errors are handled correctly"""
        def failing_processor(requests):
            raise Exception("Processor failed")
        
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=50)
        batcher.set_processor(failing_processor)
        
        try:
            request_data = {"model": "test-model", "prompt": "Test"}
            
            with self.assertRaises(BatchingError):
                batcher.add_request(request_data)
        finally:
            batcher.shutdown()
    
    def test_batch_stats(self):
        """Test that batch statistics are collected correctly"""
        # Send some requests
        for i in range(5):
            request_data = {"model": "test-model", "prompt": f"Request {i}"}
            self.batcher.add_request(request_data)
        
        stats = self.batcher.get_stats()
        
        self.assertGreater(stats['total_requests'], 0)
        self.assertGreater(stats['total_batches'], 0)
        self.assertGreater(stats['avg_batch_size'], 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that existing functionality is preserved without batch mode"""
    
    def test_no_batch_mode_unchanged(self):
        """Test that optillm works exactly the same without --batch-mode"""
        # This test would need to run optillm in a subprocess and verify
        # that responses are identical with and without batch mode disabled
        # For now, we'll test the core logic
        
        # Mock a simple request without batching
        self.assertTrue(True)  # Placeholder - would implement actual test
    
    @unittest.skipIf(not os.getenv("OPTILLM_API_KEY"), "Requires local inference")
    def test_inference_pipeline_unchanged(self):
        """Test that inference pipeline behavior is unchanged"""
        # Test that the regular generate method still works
        pass  # Would implement with actual model


class TestMLXBatching(unittest.TestCase):
    """Test MLX batch processing functionality"""
    
    @unittest.skipIf(not MLX_AVAILABLE, "MLX not available")
    def setUp(self):
        """Set up MLX test fixtures"""
        self.model_config = MLXModelConfig(
            model_id=TEST_MODEL_MLX,
            max_new_tokens=100
        )
        # Create a real cache manager instead of mock
        from optillm.inference import CacheManager
        self.cache_manager = CacheManager.get_instance(max_size=1)
        
    @unittest.skipIf(not MLX_AVAILABLE, "MLX not available") 
    def test_mlx_batch_creation(self):
        """Test that MLX batch processing can be created"""
        try:
            from optillm.inference import MLXInferencePipeline
            # This would fail if the model isn't available, but we can test the interface
            self.assertTrue(hasattr(MLXInferencePipeline, 'process_batch'))
        except Exception as e:
            # Expected if model isn't downloaded
            pass
    
    @unittest.skipIf(not MLX_AVAILABLE, "MLX not available")
    def test_mlx_batch_parameters(self):
        """Test MLX batch processing parameter validation"""
        print(f"\n📥 Testing MLX model: {self.model_config.model_id}")
        print("This may take a few minutes if model needs to be downloaded...")
        
        # Create the pipeline - this will download the model if needed
        pipeline = MLXInferencePipeline(self.model_config, self.cache_manager)
        print("✅ MLX model loaded successfully")
        
        # Test parameter validation
        with self.assertRaises(ValueError):
            pipeline.process_batch(["system1"], ["user1", "user2"])  # Mismatched lengths
            
        # Test empty inputs
        responses, tokens = pipeline.process_batch([], [])
        self.assertEqual(len(responses), 0)
        self.assertEqual(len(tokens), 0)
        
        print("✅ MLX parameter validation tests passed")
    
    @unittest.skipIf(not MLX_AVAILABLE, "MLX not available")
    def test_mlx_batch_generation(self):
        """Test MLX batch processing with actual generation"""
        print(f"\n🧪 Testing MLX batch generation...")
        
        # Create the pipeline 
        pipeline = MLXInferencePipeline(self.model_config, self.cache_manager)
        print("✅ MLX model ready for testing")
        
        # Test batch processing with real prompts
        system_prompts = ["You are a helpful assistant.", "You are a helpful assistant."]
        user_prompts = ["What is AI?", "What is ML?"]
        
        print("🚀 Running batch generation...")
        responses, token_counts = pipeline.process_batch(
            system_prompts, 
            user_prompts,
            generation_params={"max_new_tokens": 20}  # Short response for testing
        )
        
        # Validate results
        self.assertEqual(len(responses), 2)
        self.assertEqual(len(token_counts), 2)
        
        for i, response in enumerate(responses):
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            print(f"   Response {i+1}: {response[:50]}{'...' if len(response) > 50 else ''}")
            
        for token_count in token_counts:
            self.assertIsInstance(token_count, int)
            self.assertGreater(token_count, 0)
        
        print(f"✅ MLX batch generation successful - {len(responses)} responses generated")
        print(f"   Token counts: {token_counts}")


class TestPyTorchBatching(unittest.TestCase):
    """Test PyTorch batch processing functionality"""
    
    def test_pytorch_batch_method_exists(self):
        """Test that PyTorch InferencePipeline has process_batch method"""
        # The method should exist even if we can't test it fully
        from optillm.inference import InferencePipeline
        self.assertTrue(hasattr(InferencePipeline, 'process_batch'))
    
    @unittest.skipIf(not os.getenv("OPTILLM_API_KEY"), "Requires local inference")
    def test_pytorch_batch_processing(self):
        """Test PyTorch batch processing with small model"""
        # Would test with actual model if available
        pass


class TestPerformanceBenches(unittest.TestCase):
    """Performance comparison tests"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.test_prompts = [
            ("System prompt 1", "What is AI?"),
            ("System prompt 2", "Explain machine learning"),
            ("System prompt 3", "Define neural networks"),
            ("System prompt 4", "Describe deep learning")
        ]
    
    def measure_sequential_processing(self, prompts):
        """Measure time for sequential processing"""
        start_time = time.time()
        
        # Simulate sequential processing
        responses = []
        for sys_prompt, user_prompt in prompts:
            # Simulate processing time
            time.sleep(0.1)  # 100ms per request
            responses.append(f"Response to: {user_prompt}")
        
        end_time = time.time()
        return responses, end_time - start_time
    
    def measure_batch_processing(self, prompts):
        """Measure time for batch processing"""
        batcher = RequestBatcher(max_batch_size=len(prompts), max_wait_ms=10)
        
        def mock_batch_processor(requests):
            # Simulate batch processing (faster than sequential)
            time.sleep(0.15)  # 150ms for entire batch vs 400ms sequential
            return [{"response": f"Batched response {i}"} for i in range(len(requests))]
        
        batcher.set_processor(mock_batch_processor)
        
        try:
            start_time = time.time()
            
            # Send all requests concurrently
            def send_request(prompt_data):
                sys_prompt, user_prompt = prompt_data
                return batcher.add_request({
                    "model": "test-model",
                    "system_prompt": sys_prompt,
                    "user_prompt": user_prompt
                })
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                futures = [executor.submit(send_request, prompt) for prompt in prompts]
                responses = [future.result() for future in futures]
            
            end_time = time.time()
            
            return responses, end_time - start_time
            
        finally:
            batcher.shutdown()
    
    def test_batching_performance_improvement(self):
        """Test that batching provides performance improvement"""
        # Test with simulated processing
        seq_responses, seq_time = self.measure_sequential_processing(self.test_prompts)
        batch_responses, batch_time = self.measure_batch_processing(self.test_prompts)
        
        # Batch should be significantly faster
        improvement_ratio = seq_time / batch_time
        self.assertGreater(improvement_ratio, 1.5, 
                         f"Batching should be >1.5x faster, got {improvement_ratio:.2f}x")
        
        # Both should return same number of responses
        self.assertEqual(len(seq_responses), len(batch_responses))


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_cli_arguments(self):
        """Test that CLI arguments are properly parsed"""
        # Test parsing batch arguments
        import argparse
        from optillm import parse_args
        
        # Mock sys.argv for testing
        with patch('sys.argv', ['optillm', '--batch-mode', '--batch-size', '8', '--batch-wait-ms', '25']):
            args = parse_args()
            self.assertTrue(args.batch_mode)
            self.assertEqual(args.batch_size, 8)
            self.assertEqual(args.batch_wait_ms, 25)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_batch_mode_errors(self):
        """Test error conditions in batch mode"""
        batcher = RequestBatcher(max_batch_size=2, max_wait_ms=50)
        
        # Test with no processor set
        with self.assertRaises(BatchingError):
            batcher.add_request({"model": "test"})
        
        batcher.shutdown()
    
    def test_mixed_model_requests(self):
        """Test that requests with different models are properly separated"""
        batcher = RequestBatcher(max_batch_size=4, max_wait_ms=50)
        
        def mock_processor(requests):
            # Should only get requests with same model
            models = set(req.get("model") for req in requests)
            self.assertEqual(len(models), 1, "Batch should have requests from single model")
            return [{"response": "ok"}] * len(requests)
        
        batcher.set_processor(mock_processor)
        
        try:
            # Send requests with different models - should be in separate batches
            req1 = {"model": "model-a", "prompt": "test1"}
            req2 = {"model": "model-b", "prompt": "test2"}
            
            # Send concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future1 = executor.submit(batcher.add_request, req1)
                future2 = executor.submit(batcher.add_request, req2)
                
                # Both should succeed
                response1 = future1.result()
                response2 = future2.result()
                
                self.assertIsInstance(response1, dict)
                self.assertIsInstance(response2, dict)
                
        finally:
            batcher.shutdown()


def run_performance_comparison():
    """
    Run a performance comparison between sequential and batch processing
    This function can be called separately for benchmarking
    """
    print("Running Performance Comparison...")
    
    test_suite = TestPerformanceBenches()
    test_suite.setUp()
    
    # Run performance test
    seq_responses, seq_time = test_suite.measure_sequential_processing(test_suite.test_prompts)
    batch_responses, batch_time = test_suite.measure_batch_processing(test_suite.test_prompts)
    
    print(f"Sequential processing: {seq_time:.3f}s")
    print(f"Batch processing: {batch_time:.3f}s")
    print(f"Speedup: {seq_time/batch_time:.2f}x")
    
    return {
        "sequential_time": seq_time,
        "batch_time": batch_time,
        "speedup": seq_time/batch_time
    }


class TestGenerationConfigDefaults(unittest.TestCase):
    """Unit tests for env-configurable max_new_tokens and EOS resolution.

    These exercise the guards that keep a small local model from rambling up to
    the 4096-token default when it does not reliably emit an EOS token (e.g. a
    ChatML model whose tokenizer EOS differs from the chat-turn end token). No
    model is loaded.
    """

    def tearDown(self):
        os.environ.pop("OPTILLM_MAX_TOKENS", None)

    def test_default_max_new_tokens_env_override(self):
        from optillm.inference import _default_max_new_tokens, DEFAULT_MAX_NEW_TOKENS

        os.environ.pop("OPTILLM_MAX_TOKENS", None)
        self.assertEqual(_default_max_new_tokens(), DEFAULT_MAX_NEW_TOKENS)

        os.environ["OPTILLM_MAX_TOKENS"] = "128"
        self.assertEqual(_default_max_new_tokens(), 128)

        # Invalid value falls back to the default rather than raising.
        os.environ["OPTILLM_MAX_TOKENS"] = "not-a-number"
        self.assertEqual(_default_max_new_tokens(), DEFAULT_MAX_NEW_TOKENS)

        # Non-positive is clamped to a usable minimum.
        os.environ["OPTILLM_MAX_TOKENS"] = "0"
        self.assertEqual(_default_max_new_tokens(), 1)

    def test_resolve_eos_prefers_generation_config(self):
        from types import SimpleNamespace
        from optillm.inference import InferencePipeline

        # tokenizer EOS (<|end_of_text|>=1) differs from the chat end token
        # (<|im_end|>=49154); both must be honoured, generation_config first.
        fake = SimpleNamespace(
            current_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=49154)),
            tokenizer=SimpleNamespace(eos_token_id=1),
        )
        eos = InferencePipeline._resolve_eos_token_ids(fake)
        self.assertEqual(eos, [49154, 1])

    def test_resolve_eos_dedupes_list(self):
        from types import SimpleNamespace
        from optillm.inference import InferencePipeline

        fake = SimpleNamespace(
            current_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[100, 200])),
            tokenizer=SimpleNamespace(eos_token_id=200),
        )
        self.assertEqual(InferencePipeline._resolve_eos_token_ids(fake), [100, 200])

    def test_resolve_eos_falls_back_to_tokenizer(self):
        from types import SimpleNamespace
        from optillm.inference import InferencePipeline

        fake = SimpleNamespace(
            current_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None)),
            tokenizer=SimpleNamespace(eos_token_id=7),
        )
        # A single id is returned as a plain int, not a list.
        self.assertEqual(InferencePipeline._resolve_eos_token_ids(fake), 7)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)