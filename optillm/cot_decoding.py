import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def calculate_confidence(logits: List[torch.Tensor], answer_ids: torch.Tensor) -> float:
    """
    Calculate the confidence score (Δ) as specified in the paper.
    
    Args:
        logits: List of logits for each decoding step
        answer_ids: Tensor of token ids for the answer
    
    Returns:
        Confidence score (Δ)
    """
    confidence_sum = 0.0
    valid_tokens = 0
    for t, token_id in enumerate(answer_ids):
        if t >= len(logits):
            break
        token_logits = logits[t]
        probs = torch.softmax(token_logits, dim=-1)
        if probs.size(-1) > 1:
            top_2_probs, _ = torch.topk(probs, min(2, probs.size(-1)))
            if top_2_probs.size(-1) > 1:
                confidence_sum += (top_2_probs[-1][0] - top_2_probs[-1][1]).item()
            else:
                confidence_sum += 1.0  # Max confidence if there's only one token
        else:
            confidence_sum += 1.0  # Max confidence if there's only one token
        valid_tokens += 1
    
    return confidence_sum / valid_tokens if valid_tokens > 0 else 0.0

def aggregate_paths_based_on_scores(paths: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Aggregate multiple paths based on their confidence scores."""
    answer_scores = {}
    for answer, delta in paths:
        answer_scores[answer] = answer_scores.get(answer, 0) + delta
    best_answer = max(answer_scores, key=answer_scores.get)
    return best_answer, answer_scores[best_answer]

def cot_decode(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    k: int = 10,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    early_stopping: bool = False,
    aggregate_paths: bool = False,
) -> Tuple[str, float]:
    """
    Implement CoT-decoding for a given chat input.
    
    Args:
        model: The Hugging Face transformer model.
        tokenizer: The associated tokenizer.
        messages: List of chat messages in the format [{"role": "user", "content": "..."}]
        k: The number of alternative tokens to consider at the first step.
        num_beams: Number of beams for beam search.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        repetition_penalty: Repetition penalty factor.
        length_penalty: Length penalty factor.
        no_repeat_ngram_size: Size of n-grams to avoid repeating.
        early_stopping: Whether to stop generation when all beams are finished.
        aggregate_paths: Whether to aggregate multiple paths.

    Returns:
        A tuple containing the best path (or aggregated result) and its confidence score.
    """
    device = get_device()
    model.to(device)

    # Use the chat template to format the input
    if tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback for tokenizers without chat templates
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        input_text += "\nassistant:"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Set pad_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get the top-k tokens for the first decoding step
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        first_token_logits = outputs.logits[0, -1, :]
        top_k_logits, top_k_indices = torch.topk(first_token_logits, k)

    paths = []
    for idx in top_k_indices:
        # Generate sequence starting with the selected token
        start_ids = torch.cat([input_ids, idx.unsqueeze(0).unsqueeze(0)], dim=-1)
        start_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)
        
        output = model.generate(
            start_ids,
            attention_mask=start_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        
        generated_sequence = output.sequences[0]
        answer_ids = generated_sequence[len(input_ids[0]):]
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        
        # Calculate confidence score (Δ)
        confidence = calculate_confidence(output.scores, answer_ids)
        paths.append((answer_text, confidence))
    
    if aggregate_paths:
        return aggregate_paths_based_on_scores(paths)
    else:
        return max(paths, key=lambda x: x[1])
    
# Usage example
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# messages = [
#     {"role": "user", "content": "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"}
# ]

# # Generate the response using CoT decoding
# print(f"Using device: {get_device()}")
# result, confidence = cot_decode(model, tokenizer, messages, aggregate_paths=True, max_new_tokens=512)
# print(f"CoT Decoding:\n {result}")