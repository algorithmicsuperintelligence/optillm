import argparse
import json
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import logging
from openai import OpenAI

# Add parent directory to path to import optillm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_utils import TEST_MODEL

from optillm.litellm_wrapper import LiteLLMWrapper
from optillm.mcts import chat_with_mcts
from optillm.bon import best_of_n_sampling
from optillm.moa import mixture_of_agents
from optillm.rto import round_trip_optimization
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.pvg import inference_time_pv_game
from optillm.z3_solver import Z3SymPySolverSystem
from optillm.rstar import RStar
from optillm.cot_reflection import cot_reflection
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.reread import re2_approach
from optillm.mars import multi_agent_reasoning_system
from optillm.cepo.cepo import cepo, CepoConfig, init_cepo_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration - default to local inference for testing
API_KEY = os.environ.get("OPENAI_API_KEY", "optillm")

# Mock OpenAI client for testing purposes
class MockOpenAIClient:
    def chat_completions_create(self, *args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': 'Mock response'})()})]
        return MockResponse()

# Configuration for approaches
APPROACHES = {
    'mcts': chat_with_mcts,
    'bon': best_of_n_sampling,
    'moa': mixture_of_agents,
    'rto': round_trip_optimization,
    'self_consistency': advanced_self_consistency_approach,
    'pvg': inference_time_pv_game,
    'z3': lambda s, q, c, m: Z3SymPySolverSystem(s, c, m).process_query(q),
    'rstar': lambda s, q, c, m: RStar(s, c, m).solve(q),
    'cot_reflection': cot_reflection,
    'plansearch': plansearch,
    'leap': leap,
    're2': re2_approach,
    'mars': multi_agent_reasoning_system,
    'cepo': lambda s, q, c, m: cepo(s,q,c,m,init_cepo_config({'cepo_config_file': './optillm/cepo/configs/cepo_config.yaml'})),
}

def load_test_cases(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def run_approach(approach_name: str, system_prompt: str, query: str, client, model: str) -> Dict:
    start_time = time.time()
    try:
        if approach_name == 'none':
            # Direct pass-through for 'none' approach
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                # Cap output: the small local test model does not always emit an
                # EOS token on simple prompts, so bound generation to keep tests fast.
                max_tokens=64
            )
            result = (response.choices[0].message.content, response.usage.total_tokens)
        else:
            approach_func = APPROACHES[approach_name]
            result = approach_func(system_prompt, query, client, model)
        
        end_time = time.time()
        return {
            'approach': approach_name,
            'result': result,
            'time': end_time - start_time,
            'status': 'success'
        }
    except Exception as e:
        end_time = time.time()
        logger.error(f"Error in {approach_name}: {str(e)}")
        return {
            'approach': approach_name,
            'result': str(e),
            'time': end_time - start_time,
            'status': 'error'
        }

def run_test_case(test_case: Dict, approaches: List[str], client, model: str) -> Dict:
    system_prompt = test_case['system_prompt']
    query = test_case['query']
    results = []

    with ThreadPoolExecutor() as executor:
        future_to_approach = {executor.submit(run_approach, approach, system_prompt, query, client, model): approach for approach in approaches}
        for future in as_completed(future_to_approach):
            results.append(future.result())

    return {
        'test_case': test_case,
        'results': results
    }

def run_tests(test_cases: List[Dict], approaches: List[str], client, model: str, single_test_name: str = None) -> List[Dict]:
    results = []
    for test_case in test_cases:
        if single_test_name is None or test_case['name'] == single_test_name:
            result = run_test_case(test_case, approaches, client, model)
            results.append(result)
            logger.info(f"Completed test case: {test_case['name']}")
        if single_test_name and test_case['name'] == single_test_name:
            break
    return results

def print_summary(results: List[Dict]):
    print("\n=== Test Results Summary ===")
    for test_result in results:
        print(f"\nTest Case: {test_result['test_case']['name']}")
        for approach_result in test_result['results']:
            status = "✅" if approach_result['status'] == 'success' else "❌"
            print(f"  {status} {approach_result['approach']}: {approach_result['time']:.2f}s")
            if approach_result['status'] == 'error':
                print(f"     Error: {approach_result['result']}")

def main():
    parser = argparse.ArgumentParser(description="Test different LLM inference approaches.")
    parser.add_argument("--test_cases", type=str, default=None, help="Path to test cases JSON file")
    parser.add_argument("--approaches", nargs='+', default=list(APPROACHES.keys()), help="Approaches to test")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for testing")
    parser.add_argument("--base-url", type=str, default=None, help="The base_url for the OpenAI API compatible endpoint")
    parser.add_argument("--single-test", type=str, default=None, help="Name of a single test case to run")
    args = parser.parse_args()
    
    # Set default test_cases path relative to this script
    if args.test_cases is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.test_cases = os.path.join(script_dir, "test_cases.json")
    
    # If using local inference mode, override model to a local model
    if API_KEY == "optillm" and args.model == "gpt-4o-mini":
        args.model = TEST_MODEL
        logger.info(f"Using local model: {args.model}")
    
    # Set environment variable for local inference
    if API_KEY == "optillm":
        os.environ["OPTILLM_API_KEY"] = "optillm"

    test_cases = load_test_cases(args.test_cases)

    # Use local inference by default for testing
    if args.base_url:
        client = OpenAI(api_key=API_KEY, base_url=args.base_url)
    elif API_KEY == "optillm":
        # Use local inference endpoint
        client = OpenAI(api_key=API_KEY, base_url="http://localhost:8000/v1")
        logger.info("Using local inference endpoint: http://localhost:8000/v1")
    else:
        client = OpenAI(api_key=API_KEY)
        # client = LiteLLMWrapper()

    results = run_tests(test_cases, args.approaches, client, args.model, args.single_test)
    print_summary(results)

    # Optionally, save detailed results to a file
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()