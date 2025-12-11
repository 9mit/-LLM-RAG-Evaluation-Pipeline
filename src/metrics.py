import time
import torch
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class LLMEvaluator:
    def __init__(self):
        # Auto-detect hardware (Use GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Local Models on: {self.device}...")
        
        # 1. Relevance Model (Bi-Encoder)
        # Fast, lightweight model for Semantic Similarity
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # 2. Hallucination Model (Cross-Encoder / NLI)
        # Predicts if Context 'Entails' (Supports) the Response
        self.nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base', device=self.device)

    def calc_relevance(self, query: str, response: str) -> float:
        """
        Calculates how semantically similar the response is to the query.
        Returns: 0.0 to 1.0
        """
        embeddings = self.embed_model.encode([query, response])
        # Cosine similarity
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(max(0.0, min(1.0, score)))

    def calc_faithfulness(self, context: str, response: str) -> float:
        """
        Checks if the Response is grounded in the Context.
        Returns: 0.0 (Hallucination) to 1.0 (Faithful)
        """
        if not context or not response:
            return 0.0
        
        # Prepare pair for NLI model. Truncate context to 512 chars for speed/safety.
        # In a full prod system, we would chunk this intelligently.
        input_pair = [(context[:1000], response)]
        
        # Predict logits [Contradiction, Entailment, Neutral]
        scores = self.nli_model.predict(input_pair)
        
        # Convert logits to probabilities using Softmax
        probs = self._softmax(scores[0])
        
        # For 'nli-distilroberta-base', index 1 is usually 'Entailment'
        # We sum Entailment (1) and Neutral (2) as "Not Contradictory"
        # Index 0 is Contradiction.
        contradiction_score = probs[0]
        faithfulness_score = 1.0 - contradiction_score
        
        return float(faithfulness_score)

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()