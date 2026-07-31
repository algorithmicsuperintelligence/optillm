#!/usr/bin/env python3
import random
from typing import List, Dict, Any
import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import numpy as np
from huggingface_hub import HfApi

# Configure random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
DATASET_NAME = "codelion/optillmbench"
NUM_SAMPLES = 500  # Total samples in the benchmark
SPLIT_RATIO = {"train": 0.8, "test": 0.2}  # 80-20 split
SOURCES = {
    "gsm8k": {
        "name": "gsm8k",
        "subset": "main",
        "samples": 125,
        "field_map": {
            "question": "question",
            "answer": "answer"
        }
    },
    "boolq": {
        "name": "boolq",
        "subset": None,
        "samples": 125,
        "field_map": {
            "question": "question",
            "passage": "passage",
            "answer": "answer"
        }
    },
    "mmlu_math": {
        "name": "cais/mmlu",
        "subset": "all", 
        "samples": 125,
        "field_map": {
            "question": "question",
            "choices": "choices",
            "answer": "answer"
        }
    },
    "aqua_rat": {
        "name": "aqua_rat",
        "subset": None,
        "samples": 125,
        "field_map": {
            "question": "question",
            "answer": "correct"
        }
    }
}

def select_challenging_examples(
    dataset: datasets.Dataset,
    category: str,
    num_samples: int,
    field_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Select challenging examples from the dataset"""
    examples = []
    
    # Get all available examples
    all_examples = dataset["train"] if "train" in dataset else dataset["validation"]
    
    # Shuffle to randomize selection
    shuffled_indices = list(range(len(all_examples)))
    random.shuffle(shuffled_indices)
    
    # Select examples
    for idx in shuffled_indices:
        example = all_examples[idx]
        
        try:
            if category == "gsm8k":
                question = str(example[field_map["question"]])
                answer = str(example[field_map["answer"]])
                # Select only multi-step problems
                if answer.count("=") < 3:
                    continue

            elif category == "boolq":
                passage = str(example[field_map["passage"]])
                q = str(example[field_map["question"]])
                question = f"Context: {passage}\nQuestion: {q}"
                answer = "Yes" if example[field_map["answer"]] else "No"

            elif category == "mmlu_math":
                question = str(example[field_map["question"]])
                choices = example[field_map["choices"]]
                answer_index = int(example[field_map["answer"]])  # Convert answer to integer
                
                # Ensure answer index is within bounds
                if 0 <= answer_index < len(choices):
                    answer = choices[answer_index]
                else:
                    print(f"Warning: Answer index '{answer_index}' is out of range for choices: {choices}")
                    continue  # Skip this example if answer index is invalid

                # Format choices
                choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
                question = f"{question}\nChoices:\n{choices_text}"

            elif category == "aqua_rat":
                question = str(example[field_map["question"]])
                answer = str(example[field_map["answer"]])
                # Ensure non-trivial multiple-choice math problems
                if len(question.split()) < 12:
                    continue

            # General filtering
            if len(question.split()) < 10:  # Ensure substantial questions
                continue
                
            examples.append(format_question(category, question, answer))
            
            if len(examples) >= num_samples:
                break
                
        except Exception as e:
            print(f"Error processing example from {category}: {str(e)}")
            continue
            
    return examples


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing newlines"""
    return " ".join(text.replace("\r", "\n").split())

def format_question(category: str, question: str, answer: str) -> Dict[str, Any]:
    """Format a question for the benchmark dataset"""
    # Basic sanity checks
    if not question or not answer:
        raise ValueError(f"Empty question or answer in {category}")
        
    return {
        "id": f"{category}_{random.getrandbits(32):08x}",
        "category": category,
        "question": clean_text(question),
        "answer": clean_text(answer),
        "metadata": {
            "source": SOURCES[category]["name"],
            "type": category,
            "difficulty": "challenging"  # All examples are chosen to be challenging
        }
    }

def load_source_dataset(config: Dict[str, Any]) -> datasets.Dataset:
    """Load a source dataset with error handling"""
    try:
        dataset = datasets.load_dataset(
            config["name"],
            config.get("subset")
        )
        return dataset
    except Exception as e:
        print(f"Error loading dataset {config['name']}: {str(e)}")
        return None

def create_benchmark_dataset() -> Dataset:
    """Create the complete benchmark dataset"""
    all_examples = []
    
    # Process each source dataset
    for category, config in tqdm(SOURCES.items(), desc="Processing datasets"):
        print(f"\nProcessing {category} dataset...")
        
        # Load dataset
        dataset = load_source_dataset(config)
        if not dataset:
            continue
            
        # Select examples
        try:
            examples = select_challenging_examples(
                dataset,
                category,
                config["samples"],
                config["field_map"]
            )
            print(f"Selected {len(examples)} examples from {category}")
            all_examples.extend(examples)
        except Exception as e:
            print(f"Error selecting examples from {category}: {str(e)}")
            continue
    
    # Shuffle final dataset
    random.shuffle(all_examples)
    
    # Create train/test splits
    num_train = int(len(all_examples) * SPLIT_RATIO["train"])
    train_examples = all_examples[:num_train]
    test_examples = all_examples[num_train:]
    
    # Convert to HuggingFace Dataset
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_examples),
        "test": Dataset.from_list(test_examples)
    })
    
    return dataset_dict

def push_to_hub(dataset: DatasetDict, repo_id: str):
    """Push the dataset to HuggingFace Hub"""
    try:
        # Create README content
        readme_content = f"""# OptiLLMBench Dataset

A benchmark dataset for evaluating test-time optimization and scaling capabilities of language models.

## Dataset Description

OptiLLMBench contains {NUM_SAMPLES} carefully selected challenging problems across multiple domains:
- Mathematical reasoning (from competition_math)
- Code generation (from HumanEval)
- Word problems (from GSM8K)
- Multiple choice reasoning (from MMLU)
- Logical deduction (from BBH)

Each example is chosen to benefit from test-time optimization techniques like:
- Increased context length
- Chain-of-thought reasoning
- Self-consistency
- Multiple solution attempts
- And other scaling approaches

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("codelion/optillmbench")

# Access examples
for example in dataset["train"]:
    print(f"Category: {{example['category']}}")
    print(f"Question: {{example['question']}}")
    print(f"Answer: {{example['answer']}}")
    print(f"Metadata: {{example['metadata']}}")
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@software{{optillm,
  title = {{Optillm: Optimizing inference proxy for LLMs}},
  author = {{Asankhaya Sharma}},
  year = {{2024}},
  publisher = {{GitHub}},
  url = {{https://github.com/codelion/optillm}}
}}
```
"""
        
        # Push to hub
        dataset.push_to_hub(
            repo_id,
            private=False,
            embed_external_files=True
        )
        
        # Update README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        print(f"Successfully pushed dataset to {repo_id}")
        
    except Exception as e:
        print(f"Error pushing to hub: {str(e)}")

def main():
    """Main execution function"""
    print("Starting OptILM Bench dataset generation...")
    
    # Create dataset
    dataset = create_benchmark_dataset()
    
    # Print statistics
    print("\nDataset Statistics:")
    for split in dataset:
        print(f"\n{split} split:")
        print(f"Number of examples: {len(dataset[split])}")
        categories = dataset[split].unique("category")
        for category in categories:
            count = len([ex for ex in dataset[split] if ex["category"] == category])
            print(f"- {category}: {count} examples")
    
    # Push to HuggingFace Hub
    print("\nPushing dataset to HuggingFace Hub...")
    push_to_hub(dataset, DATASET_NAME)

if __name__ == "__main__":
    main()
