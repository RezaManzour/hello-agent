from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
