from dataclasses import dataclass

from pydantic import BaseModel


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


class AgentDefinition(BaseModel):
    definition: str
    confidence: float


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, object]


@dataclass
class ToolResult:
    tool: str
    content: str
    is_error: bool = False
    error: Exception | None = None

    def to_message(self) -> "Message":
        return Message(
            role="tool",
            content=self.content,
        )
