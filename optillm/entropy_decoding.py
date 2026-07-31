import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logging.info(f"Using device: {device}")

LN_2 = 0.69314718056  # ln(2)

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy

def calculate_attention_metrics(attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    attention_probs = attention_weights
    
    # Calculate entropy
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    
    # Calculate variance of entropy with unbiased=False to avoid df issues
    # Also add a check for singleton dimensions
    if attn_entropy.size(-1) > 1:
        attn_varentropy = torch.var(attn_entropy, dim=-1, unbiased=False)
    else:
        attn_varentropy = torch.zeros_like(attn_entropy)
    
    attn_varentropy = torch.where(torch.isnan(attn_varentropy), 
                                 torch.zeros_like(attn_varentropy), 
                                 attn_varentropy)
    
    # Rest remains the same
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))
    
    attention_scores_proxy = torch.log(torch.clamp(attention_probs, 1e-10, 1.0))
    interaction_strength = torch.mean(torch.abs(attention_scores_proxy), dim=(1, 2, 3))
    
    return {
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }

def _sample(logits: torch.Tensor, temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, generator: torch.Generator = None) -> torch.Tensor:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)

    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < (min_p * p_max)
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)

    top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    probs_sort = torch.flip(top_k_probs, dims=[-1])
    probs_idx = torch.flip(top_k_indices, dims=[-1])
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
    next_token = torch.multinomial(probs_sort, 1, generator=generator)
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

def adaptive_sample(logits: torch.Tensor, metrics: Dict[str, torch.Tensor],
                    gen_tokens: torch.Tensor, n_samples: int,
                    base_temp: float = 0.666, base_top_p: float = 0.90, base_top_k: int = 40, base_min_p: float = 0.03,
                    generator: torch.Generator = None) -> torch.Tensor:
    logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
    attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

    temperature = base_temp * (1 + 0.3 * logits_uncertainty + 0.2 * attn_uncertainty - 0.2 * metrics["agreement"])
    top_p = torch.clamp(base_top_p * (1 + 0.1 * metrics["attn_varentropy"]), 0.1, 1.0)
    top_k = int(torch.clamp(
        torch.round(torch.tensor(base_top_k) * (1 + 0.3 * metrics["interaction_strength"].item() - 0.2 * metrics["agreement"].item())),
        min=1,
        max=100
    ).item())
    min_p = torch.clamp(base_min_p * (1 - 0.5 * logits_uncertainty), 0.01, 0.5)

    # Convert tensor values to Python scalars for logging
    logging.debug(f"Adaptive sampling params: temp={temperature.item():.3f}, top_p={top_p.item():.3f}, top_k={top_k}, min_p={min_p.item():.3f}")

    samples = []
    for _ in range(n_samples):
        sample = _sample(logits, temperature=temperature.item(), top_p=top_p.item(), top_k=top_k, min_p=min_p.item(), generator=generator)
        samples.append(sample)

    def score_sample(sample):
        sample_flat = sample.flatten().to(torch.long)
        one_hot = F.one_hot(sample_flat, logits.shape[-1])
        log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
        log_prob = torch.sum(log_probs * one_hot)
        
        confidence_score = (
            (1 - metrics["logits_entropy"]) * 0.1 +
            (1 - metrics["attn_entropy"]) * 0.2 +
            (1 - metrics["logits_varentropy"]) * 0.3 +
            (1 - metrics["attn_varentropy"]) * 0.4 +
            metrics["agreement"] * 0.5 +
            metrics["interaction_strength"] * 0.6
        )
        return log_prob + confidence_score

    sample_scores = torch.stack([score_sample(sample) for sample in samples])
    best_sample_idx = torch.argmax(sample_scores)
    return samples[best_sample_idx]

def entropy_decode(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.666,
    top_p: float = 0.90,
    top_k: int = 27,
    min_p: float = 0.03,
    generator: torch.Generator = torch.Generator(device=device).manual_seed(1337)
) -> str:
    model.to(device)
    logging.info("Starting entropy decoding")

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        input_text += "\nassistant:"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generated_tokens = []
    gen_tokens = input_ids
    past_key_values = None
    stop = torch.tensor([tokenizer.eos_token_id], device=device, dtype=torch.int32)

    for step in range(max_new_tokens):
        logging.debug(f"Generation step: {step + 1}")
        with torch.no_grad():
            outputs = model(
                input_ids if past_key_values is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            
        logits = outputs.logits[:, -1:, :]
        attention_scores = outputs.attentions[-1]
        past_key_values = outputs.past_key_values

        entropy, varentropy = calculate_varentropy_logsoftmax(logits)
        attention_metrics = calculate_attention_metrics(attention_scores)
        metrics = {
            "logits_entropy": entropy,
            "logits_varentropy": varentropy,
            **attention_metrics
        }

        logging.debug(f"Metrics: entropy={entropy.item():.3f}, varentropy={varentropy.item():.3f}")

        if entropy < 0.1 and varentropy < 0.1:
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
            logging.debug("Using greedy sampling")
        elif entropy > 3.0 and varentropy < 0.1:
            if not torch.isin(gen_tokens[:,-1], torch.tensor([2564], device=device)).any():
                next_token = torch.tensor([[2564]], dtype=torch.int32, device=device)
                logging.debug("Inserting clarification token")
            else:
                temp_adj = 1.3 + 0.2 * attention_metrics["attn_entropy"].item()
                next_token = _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)
                logging.debug(f"Using adjusted temperature sampling: {temp_adj:.3f}")
        elif entropy < 5.0 and varentropy > 5.0:
            temp_adj = 1.2 + 0.3 * attention_metrics["interaction_strength"].item()
            top_k_adj = max(5, int(top_k * (1 + 0.5 * (1 - attention_metrics["agreement"].item()))))
            next_token = _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k_adj, min_p=min_p, generator=generator)
            logging.debug(f"Using exploration sampling: temp={temp_adj:.3f}, top_k={top_k_adj}")
        elif entropy > 5.0 and varentropy > 5.0:
            temp_adj = 2.0 + 0.5 * attention_metrics["attn_varentropy"].item()
            top_p_adj = max(0.5, top_p - 0.2 * attention_metrics["attn_entropy"].item())
            next_token = _sample(logits, temperature=max(2.0, temperature * temp_adj), top_p=top_p_adj, top_k=top_k, min_p=min_p, generator=generator)
            logging.debug(f"Using high uncertainty sampling: temp={temp_adj:.3f}, top_p={top_p_adj:.3f}")
        else:
            next_token = adaptive_sample(
                logits,
                metrics,
                gen_tokens,
                n_samples=5,
                base_temp=temperature,
                base_top_p=top_p,
                base_top_k=top_k,
                base_min_p=min_p,
                generator=generator
            )
            logging.debug("Using adaptive sampling")

        generated_tokens.append(next_token.item())
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=-1)

        logging.debug(f"Generated token: {tokenizer.decode([next_token.item()])}")

        if torch.isin(next_token, stop).any():
            logging.info("Reached stop token. Ending generation.")
            break

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logging.info("Finished entropy decoding")
    logging.info(f"Generated text: {generated_text}")

    return generated_text

# Usage example
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# messages = [
#     {"role": "user", "content": "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"}
# ]

# logging.info("Starting entropy decoding process")
# result = entropy_decode(model, tokenizer, messages)
# print(f"Entropy Decoding Result:\n{result}")
# logging.info("Entropy decoding process completed")