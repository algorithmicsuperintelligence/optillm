from typing import Tuple, Dict, Any, Optional
import logging
import outlines
import json
import torch
from pydantic import BaseModel, create_model
from transformers import AutoTokenizer, AutoModelForCausalLM

# Plugin identifier
SLUG = "json"

# Setup logging
logger = logging.getLogger(__name__)

class JSONGenerator:
    def get_device(self):
        """Get the appropriate device (mps, cuda, or cpu)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Initialize the JSON generator with a specific model."""
        self.device = self.get_device()
        logger.info(f"Using device: {self.device}")
        try:
            # Initialize the model and tokenizer using the new outlines API
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto" if str(self.device) != "cpu" else None,
                torch_dtype=torch.float16 if str(self.device) != "cpu" else torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create outlines model
            self.model = outlines.from_transformers(hf_model, self.tokenizer)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return 0

    def parse_json_schema_to_pydantic(self, schema_str: str) -> type[BaseModel]:
        """Convert JSON schema string to Pydantic model."""
        try:
            schema_dict = json.loads(schema_str)
            
            # Extract properties and required fields
            properties = schema_dict.get('properties', {})
            required = schema_dict.get('required', [])
            
            # Build field definitions for Pydantic
            fields = {}
            for field_name, field_def in properties.items():
                field_type = str  # Default to string
                
                # Map JSON schema types to Python types
                if field_def.get('type') == 'integer':
                    field_type = int
                elif field_def.get('type') == 'number':
                    field_type = float
                elif field_def.get('type') == 'boolean':
                    field_type = bool
                elif field_def.get('type') == 'array':
                    field_type = list
                elif field_def.get('type') == 'object':
                    field_type = dict
                
                # Check if field is required
                if field_name in required:
                    fields[field_name] = (field_type, ...)
                else:
                    fields[field_name] = (Optional[field_type], None)
            
            # Create dynamic Pydantic model
            return create_model('DynamicModel', **fields)
            
        except Exception as e:
            logger.error(f"Error parsing JSON schema: {str(e)}")
            raise

    def generate_json(self, prompt: str, schema: str) -> Dict[str, Any]:
        """Generate JSON based on the provided schema and prompt."""
        try:
            # Parse JSON schema to Pydantic model
            pydantic_model = self.parse_json_schema_to_pydantic(schema)
            logger.info("Parsed JSON schema to Pydantic model")

            # Generate JSON response using the new API
            result = self.model(prompt, pydantic_model)
            logger.info("Successfully generated JSON response")
            
            # Convert Pydantic model instance to dict
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            elif hasattr(result, 'dict'):
                return result.dict()
            else:
                return dict(result)

        except Exception as e:
            logger.error(f"Error generating JSON: {str(e)}")
            raise

def extract_schema_from_response_format(response_format: Dict[str, Any]) -> Optional[str]:
    """Extract schema from response_format field."""
    try:
        if not response_format:
            return None
            
        # Check if it's the OpenAI format
        if isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                schema_data = response_format.get("json_schema", {})
                if isinstance(schema_data, dict) and "schema" in schema_data:
                    return json.dumps(schema_data["schema"])
                return json.dumps(schema_data)
                
        logger.warning(f"Could not extract valid schema from response_format")
        return None
    except Exception as e:
        logger.error(f"Error extracting schema from response_format: {str(e)}")
        return None

def run(system_prompt: str, initial_query: str, client, model: str, request_config: dict = None) -> Tuple[str, int]:
    """Main plugin execution function."""
    logger.info("Starting JSON plugin execution")
    completion_tokens = 0

    try:
        # Extract schema from response_format in request_config
        response_format = request_config.get("response_format") if request_config else None
        schema = extract_schema_from_response_format(response_format)
        
        if not schema:
            logger.warning("No valid schema found in response_format")
            # Fall back to regular completion if no schema is specified
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ]
            )
            return response.choices[0].message.content, response.usage.completion_tokens

        # Initialize JSON generator
        json_generator = JSONGenerator()
        
        # Generate JSON response
        result = json_generator.generate_json(initial_query, schema)
        
        # Convert result to string if it's not already
        json_response = json.dumps(result) if isinstance(result, dict) else str(result)
        completion_tokens = json_generator.count_tokens(json_response)
        
        logger.info(f"Successfully generated JSON response: {json_response}")
        return json_response, completion_tokens

    except Exception as e:
        logger.error(f"Error in JSON plugin: {str(e)}")
        # Fall back to regular completion on error
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_query}
            ]
        )
        return response.choices[0].message.content, response.usage.completion_tokens