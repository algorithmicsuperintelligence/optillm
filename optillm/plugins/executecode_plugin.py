import re
from typing import Tuple, List
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import tempfile

SLUG = "executecode"

EXECUTE_CODE_PROMPT = '''Generate Python code to solve this problem. Put the code in a ```python block. The code:
1. Should use standard Python libraries (math, itertools, etc.)
2. Should print the final answer
3. Should be complete and runnable
4. Should include example test cases if relevant

The code will be automatically executed when submitted.'''

def extract_python_code(text: str) -> List[str]:
    """Extract Python code blocks from text."""
    # print(f"Extracting code: {text}")
    pattern = r'```python\s*(.*?)\s*```'
    return re.findall(pattern, text, re.DOTALL)

def execute_code(code: str) -> str:
    """Execute Python code in a Jupyter notebook environment."""
    
    notebook = nbformat.v4.new_notebook()
    notebook['cells'] = [nbformat.v4.new_code_cell(code)]
    
    # Convert notebook to JSON string
    notebook_json = nbformat.writes(notebook)
    
    # Convert JSON string to bytes
    notebook_bytes = notebook_json.encode('utf-8')

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.ipynb', delete=False) as tmp:
        tmp.write(notebook_bytes)
        tmp.flush()
        tmp_name = tmp.name

    try:
        with open(tmp_name, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=30, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': './'}})

        # Extract the output
        output = ""
        for cell in nb.cells:
            if cell.cell_type == 'code' and cell.outputs:
                for output_item in cell.outputs:
                    if output_item.output_type == 'stream':
                        output += output_item.text
                    elif output_item.output_type == 'execute_result':
                        output += str(output_item.data.get('text/plain', ''))
        
        return output.strip()
    finally:
        os.unlink(tmp_name)

def should_execute_request_code(query: str) -> bool:
    """Decide whether to execute code from the request based on the query."""
    keywords = ['run', 'execute', 'output', 'result']
    return any(keyword in query.lower() for keyword in keywords)

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    query, request_code = extract_python_code(initial_query)[0] if extract_python_code(initial_query) else (initial_query, "")
    
    if should_execute_request_code(query) and request_code:
        # Execute code from the request
        code_output = execute_code(request_code)
        context = f"Query: {query}\nCode:\n```python\n{request_code}\n```\nOutput:\n{code_output}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        return response.choices[0].message.content.strip(), response.usage.completion_tokens
    else:
        # Get initial response from the model
        messages = [
            {"role": "system", "content": system_prompt + EXECUTE_CODE_PROMPT} ,
            {"role": "user", "content": initial_query}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        initial_response = response.choices[0].message.content.strip()
        response_code = extract_python_code(initial_response)
        
        if response_code:
            # Execute code from the response
            code_output = execute_code(response_code[0])
            context = f"Initial response:\n{initial_response}\n\nCode output:\n{code_output}"
            
            messages.append({"role": "assistant", "content": initial_response})
            messages.append({"role": "user", "content": f"Based on the code execution output, please provide a final response:\n{context}"})
            
            final_response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            
            return final_response.choices[0].message.content.strip(), response.usage.completion_tokens + final_response.usage.completion_tokens
        else:
            return initial_response, response.usage.completion_tokens
