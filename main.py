"""CLI entry point for the LLM Evaluation Pipeline."""

from __future__ import annotations

import sys
import time

from src.loader import load_data
from src.metrics import LLMEvaluator

# Default threshold — kept consistent with dashboard.py and README (0.7)
DEFAULT_THRESHOLD = 0.7


def main() -> None:
    """Run batch evaluation over all loaded test cases."""
    print("--- Starting LLM Evaluation Pipeline (CLI) ---")

    # 1. Define paths
    chat_file = "data/chat_history.json"
    ctx_file = "data/context_data.json"

    # 2. Load data
    print(f"Loading data from {chat_file}...")
    dataset, logs = load_data(chat_file, ctx_file)

    for log in logs:
        print(f"  [LOG] {log}")

    if not dataset:
        print("Error: No data found.")
        sys.exit(1)

    # 3. Initialise models
    evaluator = LLMEvaluator()

    # 4. Run batch evaluation
    print(f"\nEvaluating {len(dataset)} test case(s)...\n")

    for i, data in enumerate(dataset):
        print(f"--- Case {i + 1} ---")
        print(f"Query: {data['query']}")

        start = time.time()

        rel_score = evaluator.calc_relevance(data["query"], data["response"])
        faith_score = evaluator.calc_faithfulness(data["context"], data["response"])

        latency_ms = (time.time() - start) * 1000

        print(f"Relevance:    {rel_score:.4f}")
        print(f"Faithfulness: {faith_score:.4f}")
        print(f"Latency:      {latency_ms:.2f} ms")

        if faith_score < DEFAULT_THRESHOLD:
            print("Status:       FAIL (Hallucination Detected)")
        else:
            print("Status:       PASS")

        print("-" * 30)


if __name__ == "__main__":
    main()