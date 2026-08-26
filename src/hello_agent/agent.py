import inspect
import json
import types
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from hello_agent.exceptions import GuardrailViolationError
from hello_agent.llm import LLMClient
from hello_agent.types import LLMResponse, Message, ToolCall, ToolResult

T = TypeVar("T", bound=BaseModel)

Tool = Callable[..., object]


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str | None = None,
        tools: dict[str, Tool] | None = None,
        max_iterations: int = 10,
        max_messages: int | None = None,
        guardrails: list[Callable[[str], bool]] | None = None,
        tool_guardrails: list[Callable[[str], bool]] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.max_messages = max_messages
        self.guardrails = guardrails or []
        self.tool_guardrails = tool_guardrails or []
        self.on_event = on_event
        self.messages: list[Message] = []

        if self.system_prompt is not None:
            self.messages.append(
                Message(
                    role="system",
                    content=self.system_prompt,
                )
            )

    @staticmethod
    def _json_type_for_annotation(annotation: object) -> str:
        json_types = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        json_type = json_types.get(annotation)

        if json_type is not None:
            return json_type

        origin = get_origin(annotation)

        if origin is list:
            item_types = get_args(annotation)

            if len(item_types) != 1:
                raise TypeError(f"Unsupported tool parameter type: {annotation}")

            return "array"

        if origin is dict:
            dict_types = get_args(annotation)

            if len(dict_types) != 2:
                raise TypeError(f"Unsupported tool parameter type: {annotation}")

            key_type, value_type = dict_types

            if key_type is not str:
                raise TypeError(f"Unsupported tool parameter type: {annotation}")

            Agent._json_type_for_annotation(value_type)

            return "object"

        if origin is types.UnionType:
            union_types = get_args(annotation)

            non_none_types = tuple(
                item for item in union_types if item is not type(None)
            )

            if len(non_none_types) == 1:
                return Agent._json_type_for_annotation(non_none_types[0])

        raise TypeError(f"Unsupported tool parameter type: {annotation}")

    @staticmethod
    def _validate_tool_argument_type(
        name: str,
        value: object,
        annotation: object,
    ) -> None:
        if annotation is inspect.Parameter.empty:
            return

        origin = get_origin(annotation)

        if origin is list:
            if not isinstance(value, list):
                raise TypeError(
                    f"Invalid type for argument '{name}': "
                    f"expected {annotation}, "
                    f"got {type(value).__name__}"
                )

            item_type = get_args(annotation)[0]

            for index, item in enumerate(value):
                Agent._validate_tool_argument_type(
                    f"{name}[{index}]",
                    item,
                    item_type,
                )

            return

        if origin is dict:
            if not isinstance(value, dict):
                raise TypeError(
                    f"Invalid type for argument '{name}': "
                    f"expected {annotation}, "
                    f"got {type(value).__name__}"
                )

            key_type, value_type = get_args(annotation)

            if key_type is not str:
                raise TypeError(f"Unsupported tool parameter type: {annotation}")

            for key, item in value.items():
                Agent._validate_tool_argument_type(
                    f"{name} key",
                    key,
                    key_type,
                )

                Agent._validate_tool_argument_type(
                    f"{name}[{key!r}]",
                    item,
                    value_type,
                )

            return

        if origin is types.UnionType:
            allowed_types = get_args(annotation)

            if not isinstance(value, allowed_types):
                raise TypeError(
                    f"Invalid type for argument '{name}': "
                    f"expected {annotation}, "
                    f"got {type(value).__name__}"
                )

            return

        if not isinstance(value, annotation):
            expected_name = getattr(
                annotation,
                "__name__",
                str(annotation),
            )

            raise TypeError(
                f"Invalid type for argument '{name}': "
                f"expected {expected_name}, "
                f"got {type(value).__name__}"
            )

    def tool_schemas(self) -> dict[str, dict]:
        schemas = {}

        for name, tool in self.tools.items():
            signature = inspect.signature(tool)
            type_hints = get_type_hints(
                tool if inspect.isroutine(tool) else tool.__call__
            )

            properties = {}
            required = []

            for parameter_name, parameter in signature.parameters.items():
                annotation = type_hints.get(
                    parameter_name,
                    parameter.annotation,
                )

                if annotation is inspect.Parameter.empty:
                    raise TypeError(
                        f"Tool parameter '{parameter_name}' "
                        "must have a type annotation."
                    )

                json_type = self._json_type_for_annotation(annotation)

                origin = get_origin(annotation)

                if origin is list:
                    item_type = get_args(annotation)[0]

                    properties[parameter_name] = {
                        "type": json_type,
                        "items": {
                            "type": Agent._json_type_for_annotation(item_type),
                        },
                    }

                elif origin is dict:
                    key_type, value_type = get_args(annotation)

                    if key_type is not str:
                        raise TypeError(
                            f"Unsupported tool parameter type: {annotation}"
                        )

                    properties[parameter_name] = {
                        "type": json_type,
                        "additionalProperties": {
                            "type": Agent._json_type_for_annotation(value_type),
                        },
                    }

                else:
                    properties[parameter_name] = {
                        "type": json_type,
                    }

                if parameter.default is inspect.Parameter.empty:
                    required.append(parameter_name)

            schemas[name] = {
                "name": name,
                "description": inspect.getdoc(tool) or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }

        return schemas

    def _execute_tool(self, tool_call: ToolCall) -> object:
        tool = self.tools.get(tool_call.tool)

        if tool is None:
            raise ValueError(f"Unknown tool: {tool_call.tool}")

        signature = inspect.signature(tool)

        try:
            bound_arguments = signature.bind(**tool_call.arguments)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

        for name, value in bound_arguments.arguments.items():
            parameter = signature.parameters[name]

            annotation = parameter.annotation

            self._validate_tool_argument_type(
                name,
                value,
                annotation,
            )

        return tool(**tool_call.arguments)

    def _emit(self, name: str, data: dict) -> None:
        if self.on_event is not None:
            self.on_event(name, data)

    def _run_tool(self, tool_call: ToolCall) -> ToolResult:
        for guardrail in self.tool_guardrails:
            for value in tool_call.arguments.values():
                if isinstance(value, str) and guardrail(value):
                    return ToolResult(
                        tool=tool_call.tool,
                        content=(
                            f"Tool error: argument blocked by guardrail "
                            f"'{guardrail.__name__}'"
                        ),
                        is_error=True,
                    )

        self._emit(
            "tool_call",
            {"tool": tool_call.tool, "arguments": tool_call.arguments},
        )

        try:
            tool_result = self._execute_tool(tool_call)
            result = ToolResult(
                tool=tool_call.tool,
                content=str(tool_result),
                is_error=False,
            )
        except Exception as exc:  # noqa: BLE001 — intentional: must catch any tool failure
            result = ToolResult(
                tool=tool_call.tool,
                content=f"Tool error: {exc}",
                is_error=True,
                error=exc,
            )

        self._emit(
            "tool_result",
            {
                "tool": result.tool,
                "is_error": result.is_error,
                "content": result.content,
            },
        )

        return result

    def save_session(self, path: str | Path) -> None:
        data = [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]

        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load_session(
        cls,
        path: str | Path,
        llm: LLMClient,
        tools: dict[str, Tool] | None = None,
        max_iterations: int = 10,
    ) -> "Agent":
        data = json.loads(Path(path).read_text())

        agent = cls(
            llm=llm,
            tools=tools,
            max_iterations=max_iterations,
        )
        agent.messages = [
            Message(role=item["role"], content=item["content"]) for item in data
        ]

        return agent

    def _enforce_message_limit(self) -> None:
        if self.max_messages is None:
            return

        if len(self.messages) <= self.max_messages:
            return

        has_system = len(self.messages) > 0 and self.messages[0].role == "system"

        if has_system:
            system_message = self.messages[0]
            remaining_slots = self.max_messages - 1
            self.messages = [system_message] + self.messages[-remaining_slots:]
        else:
            self.messages = self.messages[-self.max_messages :]

    def run(
        self,
        prompt: str,
        response_model: type[T] | None = None,
    ) -> LLMResponse | T:
        for guardrail in self.guardrails:
            if guardrail(prompt):
                raise GuardrailViolationError(
                    f"Prompt violated guardrail: {guardrail.__name__}"
                )

        self._emit("run_start", {"prompt": prompt})

        self.messages.append(
            Message(
                role="user",
                content=prompt,
            )
        )

        self._enforce_message_limit()

        for _ in range(self.max_iterations):
            tool_schemas = (
                [
                    {
                        "type": "function",
                        "function": schema,
                    }
                    for schema in self.tool_schemas().values()
                ]
                if self.tools
                else None
            )

            if response_model is not None:
                if tool_schemas is not None:
                    response = self.llm.chat(
                        self.messages.copy(),
                        response_model=response_model,
                        tools=tool_schemas,
                    )
                else:
                    response = self.llm.chat(
                        self.messages.copy(),
                        response_model=response_model,
                    )
            else:
                if tool_schemas is not None:
                    response = self.llm.chat(
                        self.messages.copy(),
                        tools=tool_schemas,
                    )
                else:
                    response = self.llm.chat(self.messages.copy())

            if not isinstance(response, ToolCall):
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

                self._enforce_message_limit()

                self._emit(
                    "run_end",
                    {
                        "content": (
                            response.content
                            if isinstance(response, LLMResponse)
                            else None
                        )
                    },
                )

                return response

            self.messages.append(
                Message(
                    role="assistant",
                    content=response.model_dump_json(),
                )
            )

            if response.tool not in self.tools:
                raise ValueError(f"Unknown tool: {response.tool}")

            tool_result = self._run_tool(response)

            if tool_result.is_error:
                self.messages.append(tool_result.to_message())

                if self.max_iterations == 1:
                    raise tool_result.error

                next_response = self.llm.chat(self.messages.copy())

                if isinstance(next_response, ToolCall):
                    if (
                        next_response.tool == response.tool
                        and next_response.arguments == response.arguments
                    ):
                        raise tool_result.error

                    response = next_response

                    self._enforce_message_limit()

                    continue

                self.messages.append(
                    Message(
                        role="assistant",
                        content=next_response.content,
                    )
                )

                self._enforce_message_limit()

                return next_response

            self.messages.append(tool_result.to_message())

            self._enforce_message_limit()

        raise RuntimeError("Agent reached the maximum number of iterations.")
