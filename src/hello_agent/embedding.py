import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from hello_agent.config import EmbeddingConfig

load_dotenv()


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.client = InferenceClient(token=os.environ["HF_TOKEN"])
        self.model = config.model

    def embed(self, text: str) -> list[float]:
        vector = self.client.feature_extraction(text, model=self.model)
        return vector.tolist()
