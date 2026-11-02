"""
Streaming Hints Plugin for OptiLLM

Monitors the model's reasoning as it is generated and injects relevant hints
from a memory bank at moments of uncertainty or between reasoning steps.
"""

from .memory_bank import Hint, HintMemoryBank
from .retriever import Injection, StreamingHintRetriever
from .streaming_hints import run_streaming_hints

__all__ = ['Hint', 'HintMemoryBank', 'Injection', 'StreamingHintRetriever', 'run_streaming_hints']
