#!/usr/bin/env python3
"""
SimpleQA Evaluation Script for OptILLM

This script evaluates OptILLM's effectiveness on the SimpleQA factuality benchmark,
with support for comparing different models and approaches (baseline, web_search, deep_research).

SimpleQA is OpenAI's factuality benchmark containing 4,326 short, fact-seeking questions
designed to be challenging for frontier models.
"""

import argparse
import json
import logging
import re
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import requests
import httpx
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
SIMPLEQA_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
SIMPLEQA_VERIFIED_CSV_URL = "https://huggingface.co/datasets/codelion/SimpleQA-Verified/raw/main/simpleqa_verified.csv"
DEFAULT_TIMEOUT = 600  # 10 minutes for potentially long research operations
DEFAULT_GRADER_MODEL = "gpt-4o"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# SimpleQA grading prompt based on OpenAI's implementation
GRADING_PROMPT = """You are grading a response to a factual question. Your task is to classify the response as:

A) CORRECT - The response contains the correct information and directly answers the question
B) INCORRECT - The response contains factual errors or contradictory information  
C) NOT_ATTEMPTED - The response doesn't attempt to answer or lacks the required information

Guidelines:
- Semantic meaning matters more than exact wording
- Minor variations in names/spelling are acceptable
- Hedging is allowed if core information is correct
- Partial answers can be correct if they contain the key fact
- Only grade as INCORRECT if there are clear factual errors

Question: {question}
Gold Answer: {gold_answer}
Response: {response}

Grade (A/B/C):"""


def remove_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from the response."""
    if not text:
        return text
    
    if '</think>' in text:
        # Get everything after the last </think> tag
        parts = text.split('</think>')
        return parts[-1].strip()
    elif '<think>' in text and '</think>' not in text:
        # Handle truncated responses (no closing tag)
        parts = text.split('<think>')
        return parts[0].strip() if len(parts) > 1 and parts[0] else ""
    
    return text


class SimpleQAEvaluator:
    """Main evaluator class for SimpleQA benchmark"""
    
    def __init__(self, 
                 model: str,
                 approach: str,
                 base_url: str = DEFAULT_BASE_URL,
                 grader_model: str = DEFAULT_GRADER_MODEL,
                 timeout: int = DEFAULT_TIMEOUT,
                 cache_dir: str = "cache",
                 output_dir: str = "results",
                 use_verified: bool = False):
        self.model = model
        self.approach = approach
        self.base_url = base_url
        self.grader_model = grader_model
        self.timeout = timeout
        self.use_verified = use_verified
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup OptILLM client with extended timeout
        self.optillm_client = OpenAI(
            api_key="optillm",
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            max_retries=0
        )
        
        # Setup grader client (use OptILLM for grading)
        try:
            self.grader_client = OpenAI(
                api_key="optillm",
                base_url=base_url,
                timeout=httpx.Timeout(timeout, connect=5.0),
                max_retries=0
            )
            logger.info("Using OptILLM for grading responses")
        except Exception as e:
            logger.warning(f"Could not initialize grader client: {e}")
            logger.warning("Grading will be skipped.")
            self.grader_client = None
        
        # Results tracking
        self.results = []
        self.metrics = {
            "correct": 0,
            "incorrect": 0, 
            "not_attempted": 0,
            "errors": 0,
            "total_processed": 0
        }
        
    def download_dataset(self) -> str:
        """Download SimpleQA dataset if not cached"""
        if self.use_verified:
            cache_file = self.cache_dir / "simpleqa_verified.csv"
            url = SIMPLEQA_VERIFIED_CSV_URL
            dataset_name = "SimpleQA-Verified"
        else:
            cache_file = self.cache_dir / "simple_qa_test_set.csv"
            url = SIMPLEQA_CSV_URL
            dataset_name = "SimpleQA"
        
        if cache_file.exists():
            logger.info(f"Using cached {dataset_name} dataset: {cache_file}")
            return str(cache_file)
        
        logger.info(f"Downloading {dataset_name} dataset from {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"Dataset downloaded to {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def load_dataset(self, num_samples: Optional[int] = None, start_index: int = 0) -> List[Dict]:
        """Load and parse SimpleQA dataset"""
        dataset_file = self.download_dataset()
        
        questions = []
        
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if i < start_index:
                        continue
                        
                    if num_samples and len(questions) >= num_samples:
                        break
                    
                    if self.use_verified:
                        # SimpleQA-Verified dataset has different fields
                        metadata = {
                            'original_index': row.get('original_index', i),
                            'topic': row.get('topic', ''),
                            'answer_type': row.get('answer_type', ''),
                            'multi_step': row.get('multi_step', ''),
                            'requires_reasoning': row.get('requires_reasoning', ''),
                            'urls': row.get('urls', '')
                        }
                        question_id = row.get('original_index', i)
                    else:
                        # Original SimpleQA dataset
                        try:
                            metadata = json.loads(row['metadata']) if row.get('metadata') else {}
                        except Exception:
                            metadata = {}
                        question_id = i
                    
                    question_data = {
                        'id': question_id,
                        'metadata': metadata,
                        'question': row['problem'],
                        'gold_answer': row['answer']
                    }
                    questions.append(question_data)
                    
            dataset_type = "SimpleQA-Verified" if self.use_verified else "SimpleQA"
            logger.info(f"Loaded {len(questions)} questions from {dataset_type} dataset")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def get_approach_config(self) -> Dict:
        """Get configuration for specific approach"""
        if self.approach == "none":
            return {}
        elif self.approach == "web_search":
            return {
                "num_results": 10,
                "headless": True,
                "timeout": 30
            }
        elif self.approach == "deep_research":
            return {
                "max_iterations": 1,
                "max_sources": 10
            }
        else:
            return {}
    
    def query_optillm(self, question: str) -> Tuple[str, bool]:
        """Query OptILLM with the specified approach"""
        try:
            # Determine model name based on approach
            if self.approach == "none":
                model_name = self.model
            else:
                model_name = f"{self.approach}-{self.model}"
            
            # Create messages
            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that provides accurate, factual answers to questions. Be direct and concise."
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            # Add approach-specific configuration
            extra_body = {}
            approach_config = self.get_approach_config()
            if approach_config:
                extra_body.update(approach_config)
            
            logger.debug(f"Querying model: {model_name}")
            logger.debug(f"Question: {question}")
            
            response = self.optillm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body=extra_body if extra_body else None,
                max_tokens=4096,
                temperature=0.6
            )
            
            answer = response.choices[0].message.content
            answer = remove_thinking_blocks(answer)
            logger.debug(f"Response: {answer}")
            
            return answer, True
            
        except Exception as e:
            logger.error(f"Error querying OptILLM: {e}")
            return f"Error: {str(e)}", False
    
    def grade_response(self, question: str, gold_answer: str, response: str) -> str:
        """Grade response using SimpleQA methodology"""
        if not self.grader_client:
            return "NOT_GRADED"
        
        try:
            grading_prompt = GRADING_PROMPT.format(
                question=question,
                gold_answer=gold_answer,
                response=response
            )
            
            grader_response = self.grader_client.chat.completions.create(
                model=self.grader_model,
                messages=[{"role": "user", "content": grading_prompt}],
                temperature=0.6,
                max_tokens=4096
            )
            
            grade_text = grader_response.choices[0].message.content.strip()
            
            # Strip <think> tags if present
            grade_text = re.sub(r'<think>.*?</think>', '', grade_text, flags=re.DOTALL).strip()
            
            # Extract grade (A/B/C)
            if grade_text.startswith('A'):
                return "CORRECT"
            elif grade_text.startswith('B'):
                return "INCORRECT"
            elif grade_text.startswith('C'):
                return "NOT_ATTEMPTED"
            else:
                logger.warning(f"Unexpected grade format: {grade_text}")
                return "NOT_GRADED"
                
        except Exception as e:
            logger.error(f"Error grading response: {e}")
            return "ERROR_GRADING"
    
    def evaluate_question(self, question_data: Dict) -> Dict:
        """Evaluate a single question"""
        question = question_data['question']
        gold_answer = question_data['gold_answer']
        
        # Query OptILLM
        response, success = self.query_optillm(question)
        
        result = {
            'id': question_data['id'],
            'metadata': question_data['metadata'],
            'question': question,
            'gold_answer': gold_answer,
            'response': response,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            # Grade the response
            grade = self.grade_response(question, gold_answer, response)
            result['grade'] = grade
            
            # Update metrics
            if grade == "CORRECT":
                self.metrics["correct"] += 1
            elif grade == "INCORRECT":
                self.metrics["incorrect"] += 1
            elif grade == "NOT_ATTEMPTED":
                self.metrics["not_attempted"] += 1
        else:
            result['grade'] = "ERROR"
            self.metrics["errors"] += 1
        
        self.metrics["total_processed"] += 1
        return result
    
    def calculate_metrics(self) -> Dict:
        """Calculate final evaluation metrics"""
        total = self.metrics["total_processed"]
        correct = self.metrics["correct"]
        incorrect = self.metrics["incorrect"]
        not_attempted = self.metrics["not_attempted"]
        errors = self.metrics["errors"]
        
        if total == 0:
            return {"error": "No questions processed"}
        
        # Basic percentages
        accuracy = (correct / total) * 100 if total > 0 else 0
        attempted = correct + incorrect
        correct_given_attempted = (correct / attempted) * 100 if attempted > 0 else 0
        
        # F1 score calculation (treating correct as TP, incorrect as FP, not_attempted as FN)
        precision = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        recall = correct / (correct + not_attempted) if (correct + not_attempted) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "total_questions": total,
            "correct": correct,
            "incorrect": incorrect, 
            "not_attempted": not_attempted,
            "errors": errors,
            "accuracy": accuracy,
            "correct_given_attempted": correct_given_attempted,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "attempted_rate": (attempted / total) * 100 if total > 0 else 0
        }
    
    def save_results(self, timestamp: str) -> Tuple[str, str, str]:
        """Save evaluation results to files"""
        # Create output directory for this run
        dataset_suffix = "_verified" if self.use_verified else ""
        run_dir = self.output_dir / f"simpleqa{dataset_suffix}_{self.model}_{self.approach}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        detailed_file = run_dir / f"{timestamp}_detailed.json"
        metrics_file = run_dir / f"{timestamp}_metrics.json"
        summary_file = run_dir / f"{timestamp}_summary.csv"
        
        # Save detailed results
        with open(detailed_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Calculate and save metrics
        final_metrics = self.calculate_metrics()
        final_metrics.update({
            "model": self.model,
            "approach": self.approach,
            "timestamp": timestamp,
            "base_url": self.base_url,
            "grader_model": self.grader_model
        })
        
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Save CSV summary
        df = pd.DataFrame(self.results)
        df.to_csv(summary_file, index=False)
        
        logger.info(f"Results saved to {run_dir}")
        
        return str(detailed_file), str(metrics_file), str(summary_file)
    
    def run_evaluation(self, 
                      num_samples: Optional[int] = None,
                      start_index: int = 0) -> Dict:
        """Run the complete evaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dataset_type = "SimpleQA-Verified" if self.use_verified else "SimpleQA"
        logger.info(f"Starting {dataset_type} evaluation")
        logger.info(f"Model: {self.model}")
        logger.info(f"Approach: {self.approach}")
        logger.info(f"Dataset: {dataset_type} ({'1k verified questions' if self.use_verified else '4.3k questions'})")
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"Timeout: {self.timeout}s")
        
        # Load dataset
        questions = self.load_dataset(num_samples, start_index)
        
        # Run evaluation with progress bar
        for question_data in tqdm(questions, desc="Evaluating questions"):
            try:
                result = self.evaluate_question(question_data)
                self.results.append(result)
                
                # Log progress periodically
                if len(self.results) % 10 == 0:
                    metrics = self.calculate_metrics()
                    logger.info(f"Progress: {len(self.results)}/{len(questions)} - "
                              f"Accuracy: {metrics['accuracy']:.1f}%")
                
            except KeyboardInterrupt:
                logger.info("Evaluation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error evaluating question {question_data['id']}: {e}")
                continue
        
        # Save results
        detailed_file, metrics_file, summary_file = self.save_results(timestamp)
        
        # Calculate final metrics
        final_metrics = self.calculate_metrics()
        
        logger.info("Evaluation completed!")
        logger.info(f"Total questions: {final_metrics['total_questions']}")
        logger.info(f"Accuracy: {final_metrics['accuracy']:.1f}%")
        logger.info(f"F1 Score: {final_metrics['f1_score']:.3f}")
        logger.info(f"Correct: {final_metrics['correct']}")
        logger.info(f"Incorrect: {final_metrics['incorrect']}")
        logger.info(f"Not Attempted: {final_metrics['not_attempted']}")
        
        return final_metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate OptILLM on SimpleQA factuality benchmark"
    )
    
    # Model and approach
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model to evaluate (default: gpt-4o-mini)")
    parser.add_argument("--approach", type=str, default="none",
                       choices=["none", "web_search", "deep_research"],
                       help="Approach to use (default: none)")
    
    # Server configuration
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                       help=f"OptILLM base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                       help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    
    # Grading configuration
    parser.add_argument("--grader-model", type=str, default=DEFAULT_GRADER_MODEL,
                       help=f"Model for grading responses (default: {DEFAULT_GRADER_MODEL})")
    
    # Evaluation parameters
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of questions to evaluate (default: all)")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Start from specific question index (default: 0)")
    
    # Search-specific parameters
    parser.add_argument("--num-search-results", type=int, default=10,
                       help="Number of search results per query (default: 10)")
    parser.add_argument("--headless", action="store_true",
                       help="Run browser in headless mode for web search")
    
    # Output configuration
    parser.add_argument("--cache-dir", type=str, default="cache",
                       help="Directory for caching dataset (default: cache)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory for saving results (default: results)")
    
    # Dataset selection
    parser.add_argument("--verified", action="store_true",
                       help="Use SimpleQA-Verified dataset (1k verified questions) instead of original SimpleQA")
    
    # Debugging
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create evaluator
    evaluator = SimpleQAEvaluator(
        model=args.model,
        approach=args.approach,
        base_url=args.base_url,
        grader_model=args.grader_model,
        timeout=args.timeout,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        use_verified=args.verified
    )
    
    try:
        # Run evaluation
        metrics = evaluator.run_evaluation(
            num_samples=args.num_samples,
            start_index=args.start_index
        )
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Approach: {args.approach}")
        print(f"Questions: {metrics['total_questions']}")
        print(f"Accuracy: {metrics['accuracy']:.1f}%")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        print(f"Correct: {metrics['correct']}")
        print(f"Incorrect: {metrics['incorrect']}")
        print(f"Not Attempted: {metrics['not_attempted']}")
        
        if metrics['errors'] > 0:
            print(f"Errors: {metrics['errors']}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()