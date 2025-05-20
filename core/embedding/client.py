from transformers import AutoModel
from enum import Enum
import numpy as np


class JinaTask(str, Enum):
    TEXT_MATCHING = "text-matching"
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_PASSAGE = "retrieval.passage"
    SEPARATION = "separation"
    CLASSIFICATION = "classification"


class JinaEmbed:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3"):
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def encode(self, texts: list[str], task: JinaTask) -> list:
        return self.model.encode(texts, task=task)

    def find_similar_texts(
        self,
        query: str,
        texts: list[str],
        task: list[JinaTask],
        top_k: int = None,
        matryoshka_dim: int = 1024,
    ) -> dict:
        query_embedding = self.encode([query], task=task[0])[0][:matryoshka_dim]

        text_embeddings = self.encode(texts, task=task[1])

        similarities = [
            float(np.dot(text_emb[:matryoshka_dim], query_embedding))
            for text_emb in text_embeddings
        ]

        results = [(i, sim, texts[i]) for i, sim in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        sorted_indices = [r[0] for r in results]
        sorted_similarities = [r[1] for r in results]
        sorted_texts = [r[2] for r in results]

        return {
            "text": sorted_texts,
            "index": sorted_indices,
            "similarity": sorted_similarities,
        }


jina_client = JinaEmbed()
