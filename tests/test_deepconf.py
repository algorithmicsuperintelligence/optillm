#!/usr/bin/env python3
"""
Simple test script for DeepConf implementation.
Tests basic functionality without requiring actual model inference.
"""

import sys
import os
import logging

# Add the optillm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all DeepConf components can be imported."""
    logger.info("Testing DeepConf imports...")
    
    try:
        from optillm.deepconf import deepconf_decode
        from optillm.deepconf.confidence import ConfidenceCalculator, ConfidenceThresholdCalibrator
        from optillm.deepconf.processor import DeepConfProcessor, TraceResult, DEFAULT_CONFIG
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_confidence_calculator():
    """Test ConfidenceCalculator functionality."""
    logger.info("Testing ConfidenceCalculator...")
    
    try:
        import torch
        from optillm.deepconf.confidence import ConfidenceCalculator
        
        calculator = ConfidenceCalculator(window_size=10, top_k=3)
        
        # Test with dummy logits
        dummy_logits = torch.randn(1000)  # Dummy logits for 1000 vocab items
        
        # Test entropy calculation
        entropy = calculator.calculate_token_entropy(dummy_logits)
        assert isinstance(entropy, float) and entropy > 0
        
        # Test confidence calculation
        confidence = calculator.calculate_token_confidence(dummy_logits)
        assert isinstance(confidence, float) and confidence > 0
        
        # Test adding tokens and group confidence
        for _ in range(15):  # Add more than window size
            calculator.add_token_confidence(dummy_logits)
        
        stats = calculator.get_trace_statistics()
        assert 'average_confidence' in stats
        assert 'num_tokens' in stats
        assert stats['num_tokens'] == 15
        
        logger.info("✓ ConfidenceCalculator tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ ConfidenceCalculator test failed: {e}")
        return False

def test_threshold_calibrator():
    """Test ConfidenceThresholdCalibrator functionality."""
    logger.info("Testing ConfidenceThresholdCalibrator...")
    
    try:
        from optillm.deepconf.confidence import ConfidenceThresholdCalibrator
        
        calibrator = ConfidenceThresholdCalibrator(variant="low")
        
        # Add some dummy confidence stats
        for i in range(5):
            stats = {
                "average_confidence": 1.0 + i * 0.1,
                "bottom_10_percent": 0.8 + i * 0.05,
                "lowest_group": 0.7 + i * 0.02
            }
            calibrator.add_warmup_trace(stats)
        
        # Test threshold calculation
        threshold = calibrator.calculate_threshold("average_confidence")
        assert isinstance(threshold, float) and threshold > 0
        
        # Test termination decision
        should_terminate = calibrator.should_terminate_trace(0.5, threshold)
        # Accept both Python bool and numpy bool
        import numpy as np
        assert isinstance(should_terminate, (bool, np.bool_))
        
        logger.info("✓ ConfidenceThresholdCalibrator tests passed")
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"✗ ConfidenceThresholdCalibrator test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_config_validation():
    """Test configuration validation."""
    logger.info("Testing configuration validation...")
    
    try:
        from optillm.deepconf.deepconf import validate_deepconf_config, DEFAULT_CONFIG
        
        # Test valid config
        valid_config = DEFAULT_CONFIG.copy()
        validated = validate_deepconf_config(valid_config)
        assert validated == valid_config
        
        # Test invalid variant
        try:
            invalid_config = {"variant": "invalid"}
            validate_deepconf_config(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test invalid numeric parameter
        try:
            invalid_config = {"warmup_samples": -1}
            validate_deepconf_config(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        logger.info("✓ Configuration validation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration validation test failed: {e}")
        return False

def test_info_function():
    """Test the info function."""
    logger.info("Testing get_deepconf_info...")
    
    try:
        from optillm.deepconf.deepconf import get_deepconf_info
        
        info = get_deepconf_info()
        
        required_keys = ["name", "description", "local_models_only", "variants", "default_config"]
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        assert info["local_models_only"]
        assert "low" in info["variants"] and "high" in info["variants"]
        
        logger.info("✓ Info function tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Info function test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting DeepConf test suite...")
    
    tests = [
        test_imports,
        test_confidence_calculator, 
        test_threshold_calibrator,
        test_config_validation,
        test_info_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! DeepConf implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)