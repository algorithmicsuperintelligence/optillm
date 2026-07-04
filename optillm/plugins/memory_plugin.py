import json
import logging
import os
import re
import tempfile
from typing import Optional, Tuple, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SLUG = "memory"

# Environment variable that opts a run in to file-backed memory. When it is set,
# the Memory store loads any previously saved items on init and persists after
# every add(); when it is unset the plugin behaves exactly as before (in-RAM,
# reset per request).
MEMORY_FILE_ENV = "OPTILLM_MEMORY_FILE"

logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, max_size: int = 100, persist_path: Optional[str] = None):
        self.max_size = max_size
        self.items: List[str] = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.completion_tokens = 0
        self.persist_path = persist_path
        if self.persist_path:
            self._load_from_file()

    def add(self, item: str):
        if len(self.items) >= self.max_size:
            self.items.pop(0)
        self.items.append(item)
        self.vectors = None  # Reset vectors to force recalculation
        if self.persist_path:
            self._save_to_file()

    def _load_from_file(self):
        """Load persisted items from ``persist_path`` (opt-in).

        A missing file (first run) or a corrupt/unreadable one degrades
        gracefully to an empty in-memory store, so persistence can never make a
        request fail at startup.
        """
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return  # nothing persisted yet
        except (OSError, ValueError) as e:
            logger.warning("Could not load memory from %s: %s", self.persist_path, e)
            return

        if not isinstance(data, list):
            logger.warning(
                "Ignoring memory file %s: expected a JSON list of strings",
                self.persist_path,
            )
            return

        # Keep only strings and honour max_size (most recent items win).
        items = [x for x in data if isinstance(x, str)]
        self.items = items[-self.max_size:]
        self.vectors = None

    def _save_to_file(self):
        """Atomically persist the current items to ``persist_path`` (opt-in).

        Writes to a temp file in the same directory and ``os.replace``s it into
        place so a crash mid-write cannot corrupt an existing store. Any I/O
        error is logged and swallowed rather than raised.
        """
        try:
            directory = os.path.dirname(os.path.abspath(self.persist_path))
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.items, f)
                os.replace(tmp_path, self.persist_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except OSError as e:
            logger.warning("Could not save memory to %s: %s", self.persist_path, e)

    def get_relevant(self, query: str, n: int = 10) -> List[str]:
        if not self.items:
            return []

        if self.vectors is None:
            self.vectors = self.vectorizer.fit_transform(self.items)

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        top_indices = similarities.argsort()[-n:][::-1]
        
        return [self.items[i] for i in top_indices]

def extract_query(text: str) -> Tuple[str, str]:
    query_index = text.rfind("Query:")
    
    if query_index != -1:
        context = text[:query_index].strip()
        query = text[query_index + 6:].strip()
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) > 1:
            context = ' '.join(sentences[:-1])
            query = sentences[-1]
        else:
            context = text
            query = "What is the main point of this text?"
    return query, context

def classify_margin(margin):
        return margin.startswith("YES#")

def extract_key_information(system_message, text: str, query: str, client, model: str) -> List[str]:
    # print(f"Prompt : {text}")
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"""
'''text
{text}
'''
Copy over all context relevant to the query: {query}
Provide the answer in the format: <YES/NO>#<Relevant context>.
Here are rules:
- If you don't know how to answer the query - start your answer with NO#
- If the text is not related to the query - start your answer with NO#
- If you can extract relevant information - start your answer with YES#
- If the text does not mention the person by name - start your answer with NO#
Example answers:
- YES#Western philosophy originated in Ancient Greece in the 6th century BCE with the pre-Socratics.
- NO#No relevant context.
"""}
    ]

    try: 
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000
        )
        key_info = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error parsing content: {str(e)}")
        return [],0
    margins = []

    if classify_margin(key_info):
        margins.append(key_info.split("#", 1)[1])
    
    return margins, response.usage.completion_tokens

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    # Opt-in file-backed memory: when OPTILLM_MEMORY_FILE is set, items persist
    # across requests; when unset, behaviour is unchanged (fresh in-RAM store).
    persist_path = os.environ.get(MEMORY_FILE_ENV) or None
    memory = Memory(persist_path=persist_path)
    query, context = extract_query(initial_query)
    completion_tokens = 0

    # Process context and add to memory
    chunk_size = 100000
    for i in range(0, len(context), chunk_size):
        chunk = context[i:i+chunk_size]
        # print(f"chunk: {chunk}")
        key_info, tokens = extract_key_information(system_prompt, chunk, query, client, model)
        #print(f"key info: {key_info}")
        completion_tokens += tokens
        for info in key_info:
            memory.add(info)
    # print(f"query : {query}")
    # Retrieve relevant information from memory
    relevant_info = memory.get_relevant(query)
    # print(f"relevant_info : {relevant_info}")
    # Generate response using relevant information
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""

I asked my assistant to read and analyse the above content page by page to help you complete this task. These are margin notes left on each page:
'''text
{relevant_info}
'''
Read again the note(s), take a deep breath and answer the query.
{query}
"""}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    # print(f"response : {response}")
    final_response = response.choices[0].message.content.strip()
    completion_tokens += response.usage.completion_tokens
    # print(f"final_response: {final_response}")
    return final_response, completion_tokens