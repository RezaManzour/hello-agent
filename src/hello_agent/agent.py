from typing import Callable, TypeVar

from pydantic import BaseModel

from hello_agent.llm import LLMClient
from hello_agent.types import LLMResponse, Message, ToolCall


T = TypeVar("T", bound=BaseModel)

Tool = Callable[..., object]


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        tools: dict[str, Tool] | None = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or {}
        self.messages: list[Message] = []

        if self.system_prompt is not None:
            self.messages.append(
                Message(
                    role="system",
                    content=self.system_prompt,
                )
            )

    def _execute_tool(self, tool_call: ToolCall) -> object:
        tool = self.tools.get(tool_call.tool)

        if tool is None:
            raise ValueError(f"Unknown tool: {tool_call.tool}")

        return tool(**tool_call.arguments)

    def run(
        self,
        prompt: str,
        response_model: type[T] | None = None,
    ) -> LLMResponse | T:
        self.messages.append(
            Message(
                role="user",
                content=prompt,
            )
        )

        if response_model is not None:
            response = self.llm.chat(
                self.messages.copy(),
                response_model=response_model,
            )
        else:
            response = self.llm.chat(self.messages.copy())

        self.messages.append(
            Message(
                role="assistant",
                content=(
                    response.model_dump_json()
                    if isinstance(response, BaseModel)
                    else response.content
                ),
            )
        )

        return response
