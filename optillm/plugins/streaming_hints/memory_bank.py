"""
Hint memory bank for the streaming hints plugin.

Stores curated hints and retrieves them by TF-IDF cosine similarity. This is
pure Python on purpose, so the plugin adds no dependencies and the bank can
later be swapped for an embedding-backed one with the same `search` interface.
"""

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_HINTS_PATH = os.path.join(os.path.dirname(__file__), "hints.json")

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have i if in is it its of on or so "
    "that the then there this to was we what when which will with you your".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


@dataclass
class Hint:
    id: str
    text: str
    keywords: List[str] = field(default_factory=list)
    domain: str = "general"
    uses: int = 0

    def index_terms(self) -> List[str]:
        # Keywords are counted twice so a curated keyword outweighs an
        # incidental word in the hint text.
        keyword_terms = tokenize(" ".join(self.keywords))
        return tokenize(self.text) + keyword_terms + keyword_terms


class HintMemoryBank:
    def __init__(self, hints: Optional[Iterable[Hint]] = None):
        self.hints: List[Hint] = []
        self._idf: Dict[str, float] = {}
        self._vectors: List[Dict[str, float]] = []
        for hint in hints or []:
            self.hints.append(hint)
        self._reindex()

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "HintMemoryBank":
        with open(path or DEFAULT_HINTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            Hint(
                id=str(item.get("id", i)),
                text=item["text"],
                keywords=list(item.get("keywords", [])),
                domain=item.get("domain", "general"),
            )
            for i, item in enumerate(data)
        )

    def add(self, hint: Hint) -> None:
        self.hints.append(hint)
        self._reindex()

    def _reindex(self) -> None:
        docs = [Counter(h.index_terms()) for h in self.hints]
        n = len(docs)
        df = Counter(term for doc in docs for term in doc)
        self._idf = {term: math.log((1 + n) / (1 + count)) + 1 for term, count in df.items()}
        self._vectors = [self._weigh(doc) for doc in docs]

    def _weigh(self, counts: Counter) -> Dict[str, float]:
        vector = {t: c * self._idf[t] for t, c in counts.items() if t in self._idf}
        norm = math.sqrt(sum(w * w for w in vector.values()))
        return {t: w / norm for t, w in vector.items()} if norm else {}

    def search(self, query: str, k: int = 1, exclude: Iterable[str] = ()) -> List[Tuple[Hint, float]]:
        """Return up to k (hint, score) pairs, best first, skipping ids in exclude."""
        query_vector = self._weigh(Counter(tokenize(query)))
        if not query_vector:
            return []
        excluded = set(exclude)
        scored = []
        for hint, vector in zip(self.hints, self._vectors):
            if hint.id in excluded:
                continue
            score = sum(w * vector.get(t, 0.0) for t, w in query_vector.items())
            if score > 0:
                scored.append((hint, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
