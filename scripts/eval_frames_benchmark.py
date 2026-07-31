import argparse
import json
import os
from typing import List, Dict

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="http://localhost:8000/v1")
# client = OpenAI()
SLEEP_INTERVAL = 300

def load_existing_results(filename: str) -> List[Dict]:
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_result(filename: str, result: Dict):
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def get_last_processed_index(results: List[Dict]) -> int:
    if not results:
        return -1
    return max(int(r.get('index', -1)) for r in results)

def generate_llm_prompt(prompt: str, wiki_links: List[str]) -> str:
    return f"Here are the relevant Wikipedia articles:\n{wiki_links}\n\nBased on all the information, answer the query. \n\nQuery: {prompt}\n\n"

def get_llm_response(prompt: str, model: str) -> str:
    response = client.with_options(timeout=1000.0).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
        extra_body={"optillm_approach": "readurls&memory"}
    )
    return response.choices[0].message.content.strip()

def evaluate_response(question: str, llm_response: str, ground_truth: str, model: str) -> Dict[str, str]:
    evaluation_prompt = f"""===Task===
I need your help in evaluating an answer provided by an LLM against a ground
truth answer. Your task is to determine if the ground truth answer is present in the LLM's
response. Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers.
Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the
"Ground Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {question}
- Predicted Answer: {llm_response}
- Ground Truth Answer: {ground_truth}
===Output Format===
Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE" )
Please proceed with the evaluation."""

    evaluation_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": evaluation_prompt}
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.3,
    )
    
    evaluation_text = evaluation_response.choices[0].message.content.strip()
    
    # Extract the decision and explanation
    lines = evaluation_text.split('\n')
    decision = "FALSE"
    explanation = ""
    for line in lines:
        if line.startswith("Decision:"):
            decision = line.split(":")[1].strip().upper()
        elif line.startswith("Explanation:"):
            explanation = line.split(":", 1)[1].strip()
    
    return {"decision": decision, "explanation": explanation}

def main(model: str):
    # Load the dataset
    dataset = load_dataset("google/frames-benchmark", split="test")
    
    filename = f"evaluation_results_{model.replace('/', '_')}.json"
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)
    
    for item in tqdm(dataset, desc="Processing samples"):
        index = int(item['Unnamed: 0'])
        if index <= last_processed_index:
            continue
        
        prompt = generate_llm_prompt(item['Prompt'], item['wiki_links'])
        llm_response = get_llm_response(prompt, model)
        evaluation = evaluate_response(item['Prompt'], llm_response, item['Answer'], model)
        
        result = {
            "index": index,
            "prompt": item['Prompt'],
            "ground_truth": item['Answer'],
            "llm_response": llm_response,
            "evaluation_decision": evaluation['decision'],
            "evaluation_explanation": evaluation['explanation'],
            "reasoning_type": item['reasoning_types']
        }
        
        save_result(filename, result)
        # print(f"Index: {index}, Decision: {result['evaluation_decision']}")
        # time.sleep(SLEEP_INTERVAL)

    # Calculate and print summary statistics
    results = load_existing_results(filename)
    total_samples = len(results)
    correct_answers = sum(1 for r in results if r['evaluation_decision'] == 'TRUE')
    accuracy = correct_answers / total_samples
    
    print(f"Model: {model}")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Print accuracy by reasoning type
    reasoning_types = set(r['reasoning_type'] for r in results)
    for rt in reasoning_types:
        rt_samples = [r for r in results if r['reasoning_type'] == rt]
        rt_correct = sum(1 for r in rt_samples if r['evaluation_decision'] == 'TRUE')
        rt_accuracy = rt_correct / len(rt_samples)
        print(f"Accuracy for {rt}: {rt_accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on google/frames-benchmark")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model to use (e.g., gpt-4o, gpt-4o-mini)")
    args = parser.parse_args()
    
    main(args.model)