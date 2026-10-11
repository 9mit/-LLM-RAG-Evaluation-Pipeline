"""Local NLP-based evaluation metrics for LLM responses.

Uses:
- ``all-MiniLM-L6-v2`` (Bi-Encoder) for semantic relevance scoring.
- ``cross-encoder/nli-distilroberta-base`` (NLI Cross-Encoder) for
  faithfulness / hallucination detection.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Model identifiers — centralised so they're easy to swap or configure
_RELEVANCE_MODEL = "all-MiniLM-L6-v2"
_NLI_MODEL = "cross-encoder/nli-distilroberta-base"

# Maximum character length of context fed to the NLI model.  Longer contexts
# are truncated for speed and to stay within model limits.
_NLI_CONTEXT_MAX_CHARS = 1000


class LLMEvaluator:
    """Compute relevance and faithfulness scores using local NLP models."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing Local Models on: %s", self.device)

        self.embed_model = SentenceTransformer(_RELEVANCE_MODEL, device=self.device)
        self.nli_model = CrossEncoder(_NLI_MODEL, device=self.device)

    def calc_relevance(self, query: str, response: str) -> float:
        """Semantic similarity between *query* and *response* (0.0–1.0)."""
        if not query or not response:
            return 0.0

        embeddings = self.embed_model.encode([query, response])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(np.clip(score, 0.0, 1.0))

    def calc_faithfulness(self, context: str, response: str) -> float:
        """Check if *response* is grounded in *context* (0.0–1.0).

        Uses NLI to measure contradiction.  Returns ``1 - P(contradiction)``.
        """
        if not context or not response:
            return 0.0

        # Truncate context to stay within model limits
        input_pair = [(context[:_NLI_CONTEXT_MAX_CHARS], response)]

        # Predict logits: [Contradiction, Entailment, Neutral]
        scores = self.nli_model.predict(input_pair)

        probs = self._softmax(scores[0])
        contradiction_score = probs[0]
        return float(1.0 - contradiction_score)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically-stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()