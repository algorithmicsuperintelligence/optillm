import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_model
from transformers import AutoTokenizer, AutoModel
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

SLUG = "router"

# Constants
MAX_LENGTH = 1024
APPROACHES = ["none", "mcts", "bon", "moa", "rto", "z3", "self_consistency", "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2"]
BASE_MODEL = "answerdotai/ModernBERT-large"
OPTILLM_MODEL_NAME = "codelion/optillm-modernbert-large"

class OptILMClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.effort_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(base_model.config.hidden_size + 64, num_labels)

    def forward(self, input_ids, attention_mask, effort):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Shape: (batch_size, hidden_size)
        effort_encoded = self.effort_encoder(effort.unsqueeze(1))  # Shape: (batch_size, 64)
        combined_input = torch.cat((pooled_output, effort_encoded), dim=1)
        logits = self.classifier(combined_input)
        return logits

# Cache the loaded router model/tokenizer/device across requests. Building it is
# expensive and one-time: it deserializes the ~400M-param ModernBERT-large base
# model, does a HuggingFace Hub round-trip to fetch the fine-tuned safetensors, and
# constructs the tokenizer. The classifier is stateless at inference time (see
# predict_approach: model.eval() + torch.no_grad), so the loaded objects are safe
# to reuse. The lock makes the first concurrent load run exactly once under the
# threaded Flask server; subsequent calls take the lock-free fast path.
_model_cache = None
_model_cache_lock = threading.Lock()


def _load_optillm_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # Load the base model
    base_model = AutoModel.from_pretrained(BASE_MODEL)
    # Create the OptILMClassifier
    model = OptILMClassifier(base_model, num_labels=len(APPROACHES))
    model.to(device)
    # Download the safetensors file
    safetensors_path = hf_hub_download(repo_id=OPTILLM_MODEL_NAME, filename="model.safetensors")
    # Load the state dict from the safetensors file
    load_model(model, safetensors_path)

    tokenizer = AutoTokenizer.from_pretrained(OPTILLM_MODEL_NAME)
    return model, tokenizer, device


def load_optillm_model():
    # Double-checked locking: return the cached bundle immediately when present,
    # and only serialize the very first (concurrent) load.
    global _model_cache
    if _model_cache is None:
        with _model_cache_lock:
            if _model_cache is None:
                _model_cache = _load_optillm_model()
    return _model_cache

def preprocess_input(tokenizer, system_prompt, initial_query):
    combined_input = f"{system_prompt}\n\nUser: {initial_query}"
    encoding = tokenizer.encode_plus(
        combined_input,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['attention_mask']

def predict_approach(model, input_ids, attention_mask, device, effort=0.7):
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        effort_tensor = torch.tensor([effort], dtype=torch.float).to(device)
        
        logits = model(input_ids, attention_mask=attention_mask, effort=effort_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_approach_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_approach_index].item()
    
    return APPROACHES[predicted_approach_index], confidence

def run(system_prompt, initial_query, client, model, **kwargs):
    try:
        # Load the trained model
        router_model, tokenizer, device = load_optillm_model()
        
        # Preprocess the input
        input_ids, attention_mask = preprocess_input(tokenizer, system_prompt, initial_query)
        
        # Predict the best approach
        predicted_approach, _ = predict_approach(router_model, input_ids, attention_mask, device)

        print(f"Router predicted approach: {predicted_approach}")
        
        # Route to the appropriate approach or use the model directly
        if predicted_approach == "none":
            # Use the model directly without routing
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ]
            )
            return response.choices[0].message.content, response.usage.completion_tokens
        elif predicted_approach == "mcts":
            return chat_with_mcts(system_prompt, initial_query, client, model, **kwargs)
        elif predicted_approach == "bon":
            return best_of_n_sampling(system_prompt, initial_query, client, model, **kwargs)
        elif predicted_approach == "moa":
            return mixture_of_agents(system_prompt, initial_query, client, model)
        elif predicted_approach == "rto":
            return round_trip_optimization(system_prompt, initial_query, client, model)
        elif predicted_approach == "z3":
            z3_solver = Z3SymPySolverSystem(system_prompt, client, model)
            return z3_solver.process_query(initial_query)
        elif predicted_approach == "self_consistency":
            return advanced_self_consistency_approach(system_prompt, initial_query, client, model)
        elif predicted_approach == "pvg":
            return inference_time_pv_game(system_prompt, initial_query, client, model)
        elif predicted_approach == "rstar":
            rstar = RStar(system_prompt, client, model, **kwargs)
            return rstar.solve(initial_query)
        elif predicted_approach == "cot_reflection":
            return cot_reflection(system_prompt, initial_query, client, model, **kwargs)
        elif predicted_approach == "plansearch":
            return plansearch(system_prompt, initial_query, client, model, **kwargs)
        elif predicted_approach == "leap":
            return leap(system_prompt, initial_query, client, model)
        elif predicted_approach == "re2":
            return re2_approach(system_prompt, initial_query, client, model, **kwargs)
        else:
            raise ValueError(f"Unknown approach: {predicted_approach}")
    
    except Exception as e:
        # Log the error and fall back to using the model directly
        print(f"Error in router plugin: {str(e)}. Falling back to direct model usage.")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}
            ]
        )
        return response.choices[0].message.content, response.usage.completion_tokens
