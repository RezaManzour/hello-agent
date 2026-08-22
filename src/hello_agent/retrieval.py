import numpy as np


class VectorStore:
    def __init__(self) -> None:
        self._texts: list[str] = []
        self._vectors: list[list[float]] = []

    def add(self, text: str, vector: list[float]) -> None:
        self._texts.append(text)
        self._vectors.append(vector)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        if not self._vectors:
            return []

        query = np.array(query_vector)
        vectors = np.array(self._vectors)

        query_norm = np.linalg.norm(query)
        vector_norms = np.linalg.norm(vectors, axis=1)

        similarities = (vectors @ query) / (vector_norms * query_norm)

        ranked_indices = np.argsort(-similarities)[:top_k]

        return [(self._texts[i], float(similarities[i])) for i in ranked_indices]


class Retriever:
    def __init__(self, embedding_client, store: VectorStore, top_k: int = 3):
        self.embedding_client = embedding_client
        self.store = store
        self.top_k = top_k

    def __call__(self, query: str) -> str:
        """Search the knowledge base for information relevant to the query."""
        query_vector = self.embedding_client.embed(query)
        results = self.store.search(query_vector=query_vector, top_k=self.top_k)
        return "\n".join(text for text, _score in results)
