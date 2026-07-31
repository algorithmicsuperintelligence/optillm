import json
import argparse
import asyncio
from tqdm import tqdm
from datasets import load_dataset
from openai import AsyncOpenAI
from typing import List, Dict, Any

# OptILM approaches remain the same as in original script
APPROACHES = ["none", "mcts", "bon", "moa", "rto", "z3", "self_consistency", "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2"]

# Dataset configurations
DATASET_CONFIGS = [
    ("MixEval", "free_form"),
    ("MixEval", "multiple_choice"),
    ("MixEval_Hard", "free_form"),
    ("MixEval_Hard", "multiple_choice")
]

def construct_prompt(sample: Dict[str, Any], split_type: str) -> str:
    """Construct prompt based on split type."""
    context = sample.get("context", "")
    prompt = sample["prompt"]
    
    if split_type == "multiple_choice":
        options = sample["options"]
        options_text = "\nOptions:\n" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        return f"Context: {context}\n\nQuestion: {prompt}{options_text}\n\nProvide the correct answer from the options above."
    else:
        return f"Context: {context}\n\nQuestion: {prompt}\n\nProvide your answer."

def is_correct_response(response: str, targets: List[str]) -> bool:
    """Check if response matches any of the target answers."""
    response = response.strip().lower()
    return any(target.strip().lower() == response for target in targets)

async def generate_response(prompt: str, approach: str) -> Dict[str, Any]:
    """Generate a response using the specified approach."""
    if approach == "none":
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }
    else:
        client = AsyncOpenAI(api_key="none", base_url="http://localhost:8000/v1")
        response = await client.chat.completions.create(
            model=f"{approach}-gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }

def rank_responses(responses: List[Dict[str, Any]], targets: List[str]) -> List[int]:
    """Rank responses based on correctness and token efficiency."""
    # Create tuples of (index, is_correct, tokens) for sorting
    ranked_data = []
    for i, response in enumerate(responses):
        is_correct = is_correct_response(response["content"], targets)
        ranked_data.append((i, is_correct, response["tokens"]))
    
    # Sort by correctness (True first) and then by tokens (ascending)
    ranked_data.sort(key=lambda x: (-int(x[1]), x[2]))
    
    # Extract indices for final ranking
    return [idx for idx, _, _ in ranked_data]

async def process_sample(sample: Dict[str, Any], split_type: str) -> Dict[str, Any]:
    """Process a single sample from the dataset."""
    prompt = construct_prompt(sample, split_type)
    results = []

    # Generate responses for each approach
    for approach in APPROACHES:
        response = await generate_response(prompt, approach)
        results.append({"approach": approach, **response})

    # Rank the responses based on correctness and token efficiency
    rankings = rank_responses(results, sample["target"])

    # Add rankings to results
    for rank, idx in enumerate(rankings):
        results[idx]["rank"] = rank

    return {
        "prompt": prompt,
        "results": results,
    }

async def generate_dataset(num_samples: int, output_file: str):
    """Generate the dataset and save it to a JSONL file."""
    with open(output_file, "w") as f:
        for config, split_type in DATASET_CONFIGS:
            print(f"Processing {config} - {split_type}")
            dataset = load_dataset("MixEval/MixEval", config, split=split_type)
            
            # Calculate samples per configuration
            samples_per_config = max(1, num_samples // len(DATASET_CONFIGS))
            
            for sample in tqdm(dataset.select(range(samples_per_config)), 
                             total=samples_per_config,
                             desc=f"{config}-{split_type}"):
                try:
                    result = await process_sample(sample, split_type)
                    f.write(json.dumps(result) + "\n")
                except Exception as e:
                    print(f"Error processing sample: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate OptILM Ground Truth dataset")
    parser.add_argument("--num_samples", type=int, default=100, 
                      help="Total number of samples to process (divided among configurations)")
    parser.add_argument("--output_file", type=str, 
                      default="optillm_ground_truth_dataset.jsonl",
                      help="Output file path")
    args = parser.parse_args()

    asyncio.run(generate_dataset(args.num_samples, args.output_file))
    print(f"Dataset generated and saved to {args.output_file}")

if __name__ == "__main__":
    main()