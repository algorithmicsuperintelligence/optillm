# Streaming Hints

Injects hints from a memory bank into the model's reasoning while it is being generated, inspired by [this idea from Will Brown](https://x.com/willccbb/status/1940557166248972387). Instead of putting every piece of guidance in the prompt up front, the plugin watches the reasoning and adds one targeted hint at the moment the model hesitates or moves to a new step.

This is the inference half of the approach. The retriever is rule-based for now; training the injection policy with RL is future work (see below).

## How it works

1. The model answers the query in segments of at most `segment_tokens` tokens.
2. Each segment is fed sentence by sentence to the `StreamingHintRetriever`.
3. A sentence is an **injection point** when at least `min_gap_chars` characters have passed since the last hint and the sentence contains an uncertainty signal ("wait", "hmm", "not sure", "let me check", ...) or ends a paragraph.
4. At an injection point the retriever queries the `HintMemoryBank` with the problem plus the most recent reasoning. If the best unused hint scores at least `hint_threshold`, the segment is cut after that sentence and the hint is appended as `<hint>...</hint>`.
5. Generation continues from the cut, with the reasoning so far (hints included) in the prompt. This repeats until the model finishes, `max_segments` is reached, or `max_tokens` is spent.

Segments are used instead of a token stream so the plugin works with every client optillm supports, including the local inference server, which does not stream. The cost is that up to one segment of text after an injection point is generated and then discarded.

The memory bank ranks hints by TF-IDF cosine similarity over the hint text and its keywords. It is pure Python, so the plugin has no extra dependencies.

## Usage

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="streaming_hints-gpt-4o-mini",
    messages=[{"role": "user", "content": "A train travels 120 km in 1.5 hours. What is its speed in m/s?"}],
    extra_body={"hint_threshold": 0.25, "show_hints": True},
)
```

It combines with other approaches like any plugin, for example `streaming_hints&cot_reflection-gpt-4o-mini`.

## Configuration

All options are read from the request body (`extra_body`).

| Option | Default | Description |
|---|---|---|
| `hint_threshold` | `0.2` | Minimum similarity score for a hint to be injected |
| `max_hints` | `3` | Maximum number of hints injected per request |
| `min_gap_chars` | `200` | Minimum characters of reasoning between two injections |
| `segment_tokens` | `512` | Maximum tokens generated per segment |
| `max_segments` | `8` | Maximum number of segments per request |
| `hints_path` | bundled `hints.json` | Path to a custom hint bank |
| `show_hints` | `false` | Return the reasoning with the `<hint>` tags instead of the model text only |

`max_tokens`, if set, caps the completion tokens across all segments.

## Custom hints

A hint bank is a JSON list. `keywords` count twice as much as words in `text` when matching.

```json
[
  {
    "id": "math-units",
    "domain": "math",
    "text": "Write down the units of every quantity and check both sides match.",
    "keywords": ["units", "speed", "rate", "convert"]
  }
]
```

The bundled bank has 20 general hints for math, coding and logical reasoning. Domain-specific banks will usually do better.

## Future work

- **RL-trained retriever.** Learn when to inject and which hint to pick from downstream task accuracy, with and without hints as the reward signal. `Hint.uses` is tracked as a starting point for usage statistics.
- **Embedding retrieval.** A bank backed by sentence embeddings can replace TF-IDF behind the same `search` interface.
- **Token streaming.** For providers that stream, the retriever can be fed token deltas directly and the stream cancelled at the injection point, which avoids the discarded tail of a segment.
- **Evaluation.** Benchmark results (GSM8K, MATH, HumanEval, MMLU) against the plain model and against putting all hints in the prompt.
