import os

from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    def __init__(self):
        self.model = os.getenv(
            "LLM_MODEL",
            "openai/gpt-oss-120b:fireworks-ai",
        )

        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))

        self.timeout = float(os.getenv("LLM_TIMEOUT", "30"))


class EmbeddingConfig:
    def __init__(self):
        self.model = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
