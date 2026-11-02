"""
Streaming hint retriever.

Consumes the model's reasoning incrementally, decides where a hint may be
injected (an uncertainty signal or a paragraph break), and retrieves the best
matching unused hint for the problem plus the most recent reasoning.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from .memory_bank import Hint, HintMemoryBank

UNCERTAINTY_PATTERN = re.compile(
    r"\b(wait|hmm+|hold on|not sure|unsure|maybe|perhaps|actually|alternatively|"
    r"on second thought|let me (?:re)?(?:check|think|consider|verify|try)|"
    r"(?:doesn'?t|does not) (?:seem|look) right|i (?:might|may) have made)\b",
    re.IGNORECASE,
)

# A sentence ends at ., ! or ? followed by whitespace, or at a newline.
_BOUNDARY = re.compile(r"[.!?](?=\s)|\n")


@dataclass
class Injection:
    hint: Hint
    score: float
    # Offset into the text passed to the feed() call that produced this
    # injection. The caller keeps text[:offset] and discards the rest.
    offset: int


class StreamingHintRetriever:
    def __init__(
        self,
        memory_bank: HintMemoryBank,
        threshold: float = 0.2,
        max_hints: int = 3,
        min_gap_chars: int = 200,
        window_chars: int = 600,
    ):
        self.memory_bank = memory_bank
        self.threshold = threshold
        self.max_hints = max_hints
        self.min_gap_chars = min_gap_chars
        self.window_chars = window_chars
        self.reset("")

    def reset(self, problem: str) -> None:
        self.problem = problem
        self.text = ""
        self.pending = ""
        self.chars_since_injection = 0
        self.used: List[str] = []

    def feed(self, delta: str) -> Optional[Injection]:
        """
        Consume the next piece of reasoning. Returns the first injection found
        in it, or None. On an injection the text after the offset is treated
        as discarded and is not kept in the retriever's state.
        """
        if len(self.used) >= self.max_hints:
            self.text += delta
            return None

        consumed = 0
        for match in _BOUNDARY.finditer(delta):
            end = match.end()
            sentence = self.pending + delta[consumed:end]
            consumed = end
            self.pending = ""
            self.text += sentence
            self.chars_since_injection += len(sentence)

            if not self._is_injection_point(sentence):
                continue
            injection = self._retrieve(end)
            if injection:
                self.chars_since_injection = 0
                return injection

        self.pending += delta[consumed:]
        return None

    def _is_injection_point(self, sentence: str) -> bool:
        if self.chars_since_injection < self.min_gap_chars:
            return False
        if UNCERTAINTY_PATTERN.search(sentence):
            return True
        # A blank line marks a transition between reasoning steps.
        return self.text.endswith("\n\n")

    def _retrieve(self, offset: int) -> Optional[Injection]:
        query = f"{self.problem}\n{self.text[-self.window_chars:]}"
        results = self.memory_bank.search(query, k=1, exclude=self.used)
        if not results or results[0][1] < self.threshold:
            return None
        hint, score = results[0]
        hint.uses += 1
        self.used.append(hint.id)
        return Injection(hint=hint, score=score, offset=offset)
