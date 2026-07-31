import json
import argparse
import asyncio
from tqdm import tqdm
from datasets import load_dataset
from openai import AsyncOpenAI
from typing import List, Dict, Any
import random

# OptILM approaches
APPROACHES = ["none", "mcts", "bon", "moa", "rto", "z3", "self_consistency", "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2"]

async def generate_response(prompt: str, approach: str) -> Dict[str, Any]:
    """Generate a response using the specified approach."""
    if approach == "none":
        # Use the base model without any optimization technique
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
        # Use OptILM with the specified approach
        client = AsyncOpenAI(api_key="none", base_url="http://localhost:8080/v1")
        response = await client.chat.completions.create(
            model=f"{approach}-gpt-4o-mini",  # Assuming OptILM uses this naming convention
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "content": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens,
        }

async def rank_responses(prompt: str, responses: List[Dict[str, Any]]) -> List[int]:
    """Rank the responses using the LLM."""
    ranking_prompt = f"Given the following prompt:\n\n{prompt}\n\nRank the following responses from best to worst, considering accuracy, completeness, and relevance. Provide the ranking as a comma-separated list of indices (0-indexed). Do not add any explanations or any other text other than the comma-separated list.\n\n"
    for i, response in enumerate(responses):
        ranking_prompt += f"Response {i}:\n{response['content']}\n\n"
    client = AsyncOpenAI()
    ranking_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ranking_prompt}],
    )
    
    ranking_str = ranking_response.choices[0].message.content.strip()
    print(f"Ranking str: {ranking_str}")
    return [int(idx) for idx in ranking_str.split(",")]

async def process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single sample from the dataset."""
    prompt = sample["turns"][0]["content"]
    results = []

    # Generate responses for each approach
    for approach in APPROACHES:
        response = await generate_response(prompt, approach)
        results.append({"approach": approach, **response})

    random.shuffle(results)
    # Rank the responses
    rankings = await rank_responses(prompt, results)

    # Add rankings to results
    print(rankings)
    for rank, idx in enumerate(rankings):
        results[idx]["rank"] = rank

    return {
        "prompt": prompt,
        "results": results,
    }

async def generate_dataset(num_samples: int, output_file: str):
    """Generate the dataset and save it to a JSONL file."""
    dataset = load_dataset("lmsys/arena-hard-auto-v0.1", split="train")
    
    with open(output_file, "w") as f:
        for sample in tqdm(dataset.select(range( num_samples)), total=num_samples):
            try:
                result = await process_sample(sample)
                f.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"Skip over this item due to error {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate OptILM dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--output_file", type=str, default="optillm_dataset.jsonl", help="Output file path")
    args = parser.parse_args()

    asyncio.run(generate_dataset(args.num_samples, args.output_file))
    print(f"Dataset generated and saved to {args.output_file}")

if __name__ == "__main__":
    main()
