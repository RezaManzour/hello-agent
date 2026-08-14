import os
from typing import TypeVar

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel

from hello_agent.config import LLMConfig
from hello_agent.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
)
from hello_agent.types import LLMResponse, Message


T = TypeVar("T", bound=BaseModel)

load_dotenv()


class LLMClient:
    def __init__(self, config: LLMConfig):
        token = os.getenv("HF_TOKEN")

        if not token:
            raise LLMAuthenticationError("HF_TOKEN is not set")

        self.client = InferenceClient(
            token=token,
            timeout=config.timeout,
        )

        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def chat(
        self,
        messages: list[Message],
        response_model: type[T] | None = None,
    ) -> LLMResponse | T:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in messages
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": response_model.__name__,
                                "schema": response_model.model_json_schema(),
                            },
                        }
                    }
                    if response_model is not None
                    else {}
                ),
            )

        except HfHubHTTPError as exc:
            status_code = exc.response.status_code

            if status_code == 401:
                raise LLMAuthenticationError(
                    "Hugging Face authentication failed."
                ) from exc

            if status_code == 429:
                raise LLMRateLimitError(
                    "Hugging Face rate limit exceeded."
                ) from exc

            raise LLMProviderError(
                f"Hugging Face provider error: HTTP {status_code}"
            ) from exc

        choice = response.choices[0]
        usage = response.usage

        if response_model is not None:
            try:
                return response_model.model_validate_json(
                    choice.message.content
                )
            except Exception as exc:
                raise LLMProviderError(
                    "Failed to parse structured LLM response."
                ) from exc

        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
