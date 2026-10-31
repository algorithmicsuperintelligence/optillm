# OptiLLM Tests

This directory contains tests for the OptiLLM project.

## Structure

- `test.py` - Main comprehensive test suite for all OptiLLM approaches
- `test_cases.json` - Test cases for the main test suite
- `test_plugins.py` - Unit tests for plugin functionality
- `test_api_compatibility.py` - Tests for OpenAI API compatibility
- `test_n_parameter.py` - Tests for n parameter functionality (multiple completions)
- `test_approaches.py` - Unit tests for approach modules (no model inference required)
- `test_ci_quick.py` - Quick CI tests for imports and basic functionality
- `run_tests.sh` - Convenience script to run all tests
- `requirements.txt` - Test dependencies (pytest, etc.)

## Running Tests

### Prerequisites

1. Install test dependencies:
   ```bash
   pip install -r tests/requirements.txt
   ```

2. Start the OptiLLM server:
   ```bash
   python optillm.py
   ```

### Run All Tests

```bash
./tests/run_tests.sh
```

### Run Specific Tests

```bash
# Unit tests only (no server required)
pytest tests/test_plugins.py

# API tests (requires running server)
pytest tests/test_api_compatibility.py

# N parameter test
python tests/test_n_parameter.py
```

### Run with pytest

```bash
# Run all tests in the tests directory
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=optillm --cov-report=html
```

## Main Test Suite

The main test suite (`test.py`) is located in the tests directory along with its test data (`test_cases.json`).

To run the main test suite from the project root:
```bash
python tests/test.py
```

Or from within the tests directory:
```bash
cd tests
python test.py
```

## CI/CD

Tests are automatically run on:
- Every push to the main branch
- Every pull request

The GitHub Actions workflow (`.github/workflows/test.yml`) runs:
1. Quick CI tests (imports and basic functionality)
2. Unit tests for plugins and approaches (no model inference required)
3. Integration tests with OpenAI API (only on PRs from same repository with secrets)

### CI Testing Strategy

To keep CI fast and reliable:
- Unit tests don't require model inference or a running server
- Integration tests only run with real API keys when available
- The main `tests/test.py` is used for comprehensive local testing
- For CI, we use simplified tests that verify structure and imports

## Writing New Tests

1. Add unit tests to appropriate files in `tests/`
2. Follow pytest conventions (test functions start with `test_`)
3. Use fixtures for common setup
4. Add integration tests that require the server to `test_api_compatibility.py`

## Test Coverage

To generate a coverage report:
```bash
pytest tests/ --cov=optillm --cov-report=html
open htmlcov/index.html
```