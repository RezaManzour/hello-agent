import pytest
from unittest.mock import Mock

from hello_agent.agent import Agent
from hello_agent.types import AgentDefinition, LLMResponse, Message, ToolCall


def test_agent_returns_llm_response():
    fake_llm = Mock()

    fake_llm.chat.return_value = LLMResponse(
        content="Hello from the agent.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    agent = Agent(llm=fake_llm)

    result = agent.run("Say hello.")

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello from the agent."

    fake_llm.chat.assert_called_once_with(
        [
            Message(
                role="user",
                content="Say hello.",
            )
        ]
    )


def test_agent_includes_system_prompt():
    fake_llm = Mock()

    fake_llm.chat.return_value = LLMResponse(
        content="Hello.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    agent = Agent(
        llm=fake_llm,
        system_prompt="You are a helpful AI assistant.",
    )

    agent.run("Say hello.")

    fake_llm.chat.assert_called_once_with(
        [
            Message(
                role="system",
                content="You are a helpful AI assistant.",
            ),
            Message(
                role="user",
                content="Say hello.",
            ),
        ]
    )


def test_agent_preserves_conversation_history():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        LLMResponse(
            content="Nice to meet you, Reza.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        LLMResponse(
            content="Your name is Reza.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    agent = Agent(
        llm=fake_llm,
        system_prompt="You are a helpful assistant.",
    )

    agent.run("My name is Reza.")
    result = agent.run("What is my name?")

    assert result.content == "Your name is Reza."

    second_call_messages = fake_llm.chat.call_args_list[1].args[0]

    assert second_call_messages == [
        Message(
            role="system",
            content="You are a helpful assistant.",
        ),
        Message(
            role="user",
            content="My name is Reza.",
        ),
        Message(
            role="assistant",
            content="Nice to meet you, Reza.",
        ),
        Message(
            role="user",
            content="What is my name?",
        ),
    ]


def test_agent_supports_structured_output():
    fake_llm = Mock()

    expected_result = AgentDefinition(
        definition="An AI agent is software that acts toward a goal.",
        confidence=0.98,
    )

    fake_llm.chat.return_value = expected_result

    agent = Agent(llm=fake_llm)

    result = agent.run(
        "Define an AI agent.",
        response_model=AgentDefinition,
    )

    assert isinstance(result, AgentDefinition)
    assert result.definition == expected_result.definition
    assert result.confidence == expected_result.confidence

    fake_llm.chat.assert_called_once_with(
        [
            Message(
                role="user",
                content="Define an AI agent.",
            )
        ],
        response_model=AgentDefinition,
    )


def test_agent_can_execute_a_tool():
    fake_llm = Mock()

    fake_llm.chat.return_value = LLMResponse(
        content="The result is 450.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    def calculator(a: int, b: int) -> int:
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 25 × 18.")

    assert result.content == "The result is 450."


def test_tool_call_model():
    from hello_agent.types import ToolCall

    tool_call = ToolCall(
        tool="calculator",
        arguments={
            "a": 25,
            "b": 18,
        },
    )

    assert tool_call.tool == "calculator"
    assert tool_call.arguments == {
        "a": 25,
        "b": 18,
    }


def test_agent_executes_tool_call():
    fake_llm = Mock()

    def calculator(a: int, b: int) -> int:
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    tool_call = ToolCall(
        tool="calculator",
        arguments={
            "a": 25,
            "b": 18,
        },
    )

    result = agent._execute_tool(tool_call)

    assert result == 450

def test_agent_executes_tool_call_and_returns_final_response():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={
                "a": 15,
                "b": 30,
            },
        ),
        LLMResponse(
            content="The result is 450.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={
            "calculator": calculator,
        },
    )

    result = agent.run("What is 15 multiplied by 30?")

    assert isinstance(result, LLMResponse)
    assert result.content == "The result is 450."

    assert fake_llm.chat.call_count == 2

    second_call_messages = fake_llm.chat.call_args_list[1].args[0]

    assert second_call_messages[0] == Message(
        role="user",
        content="What is 15 multiplied by 30?",
    )

    assert second_call_messages[1] == Message(
        role="assistant",
        content='{"tool":"calculator","arguments":{"a":15,"b":30}}',
    )

    assert second_call_messages[2] == Message(
        role="tool",
        content="450",
    )

def test_agent_executes_multiple_tool_calls():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={
                "a": 15,
                "b": 30,
            },
        ),
        ToolCall(
            tool="calculator",
            arguments={
                "a": 450,
                "b": 2,
            },
        ),
        LLMResponse(
            content="The final result is 900.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=30,
            completion_tokens=8,
            total_tokens=38,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run(
        "Calculate 15 multiplied by 30, then multiply the result by 2."
    )

    assert isinstance(result, LLMResponse)
    assert result.content == "The final result is 900."

    assert fake_llm.chat.call_count == 3

    third_call_messages = fake_llm.chat.call_args_list[2].args[0]

    assert third_call_messages[-2] == Message(
        role="assistant",
        content='{"tool":"calculator","arguments":{"a":450,"b":2}}',
    )

    assert third_call_messages[-1] == Message(
        role="tool",
        content="900",
    )

def test_agent_stops_after_max_iterations():
    fake_llm = Mock()

    fake_llm.chat.return_value = ToolCall(
        tool="calculator",
        arguments={
            "a": 1,
            "b": 1,
        },
    )

    def calculator(a: int, b: int) -> int:
        return a + b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
        max_iterations=3,
    )

    try:
        agent.run("Keep calculating.")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        assert "maximum" in str(exc).lower()

    assert fake_llm.chat.call_count == 3

def test_agent_builds_tool_schemas():
    fake_llm = Mock()

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    schemas = agent.tool_schemas()

    assert "calculator" in schemas

    calculator_schema = schemas["calculator"]

    assert calculator_schema["name"] == "calculator"
    assert calculator_schema["description"] == "Multiply two numbers."
    assert calculator_schema["parameters"]["type"] == "object"
    assert "a" in calculator_schema["parameters"]["properties"]
    assert "b" in calculator_schema["parameters"]["properties"]
    assert calculator_schema["parameters"]["required"] == ["a", "b"]


def test_agent_sends_tool_result_back_to_llm():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": 15, "b": 30},
        ),
        LLMResponse(
            content="The result is 450.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 times 30.")

    assert isinstance(result, LLMResponse)
    assert result.content == "The result is 450."

    assert fake_llm.chat.call_count == 2

    second_call_messages = fake_llm.chat.call_args_list[1].args[0]

    assert second_call_messages == [
        Message(
            role="user",
            content="Calculate 15 times 30.",
        ),
        Message(
            role="assistant",
            content='{"tool":"calculator","arguments":{"a":15,"b":30}}',
        ),
        Message(
            role="tool",
            content="450",
        ),
    ]


def test_agent_rejects_unknown_tool():
    fake_llm = Mock()

    fake_llm.chat.return_value = ToolCall(
        tool="unknown_tool",
        arguments={},
    )

    agent = Agent(
        llm=fake_llm,
        tools={},
    )

    with pytest.raises(ValueError, match="Unknown tool: unknown_tool"):
        agent.run("Use the unknown tool.")


def test_agent_propagates_tool_errors():
    fake_llm = Mock()

    fake_llm.chat.return_value = ToolCall(
        tool="calculator",
        arguments={"a": 15, "b": 0},
    )

    def calculator(a: int, b: int) -> int:
        return a // b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    with pytest.raises(ZeroDivisionError):
        agent.run("Calculate 15 divided by 0.")


def test_agent_recovers_from_tool_error():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": 15, "b": 0},
        ),
        LLMResponse(
            content="I couldn't calculate that because division by zero is not allowed.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=12,
            total_tokens=32,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        return a // b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 divided by 0.")

    assert isinstance(result, LLMResponse)
    assert "division by zero" in result.content.lower()

    assert fake_llm.chat.call_count == 2

    second_call_messages = fake_llm.chat.call_args_list[1].args[0]

    assert second_call_messages[-1] == Message(
        role="tool",
        content="Tool error: integer division or modulo by zero",
    )


def test_agent_tool_schemas_include_function_metadata():
    fake_llm = Mock()

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    schemas = agent.tool_schemas()

    assert schemas == {
        "calculator": {
            "name": "calculator",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        }
    }


def test_agent_passes_tool_schemas_to_llm():
    fake_llm = Mock()

    fake_llm.chat.return_value = LLMResponse(
        content="The result is 450.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    agent.run("Calculate 15 times 30.")

    fake_llm.chat.assert_called_once()

    call_kwargs = fake_llm.chat.call_args.kwargs

    assert call_kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Multiply two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]


def test_agent_passes_tools_on_every_iteration():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": 15, "b": 30},
        ),
        LLMResponse(
            content="The result is 450.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 times 30.")

    assert isinstance(result, LLMResponse)
    assert result.content == "The result is 450."

    assert fake_llm.chat.call_count == 2

    for call in fake_llm.chat.call_args_list:
        assert call.kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Multiply two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]


def test_agent_supports_multiple_tools():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="add",
            arguments={"a": 10, "b": 5},
        ),
        LLMResponse(
            content="The result is 15.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={
            "add": add,
            "multiply": multiply,
        },
    )

    agent.run("Add 10 and 5.")

    assert fake_llm.chat.call_count == 2

    for call in fake_llm.chat.call_args_list:
        tools = call.kwargs["tools"]

        assert len(tools) == 2

        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "add"

        assert tools[1]["type"] == "function"
        assert tools[1]["function"]["name"] == "multiply"




def test_agent_rejects_missing_tool_argument():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": 15},
        ),
        LLMResponse(
            content="I could not calculate the result.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 times 30.")

    assert isinstance(result, LLMResponse)
    assert result.content == "I could not calculate the result."

    assert fake_llm.chat.call_count == 2

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert "missing" in tool_message.content.lower()


def test_agent_rejects_extra_tool_argument():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": 15, "b": 30, "c": 100},
        ),
        LLMResponse(
            content="I could not calculate the result.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 times 30.")

    assert isinstance(result, LLMResponse)
    assert result.content == "I could not calculate the result."

    assert fake_llm.chat.call_count == 2

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert "unexpected" in tool_message.content.lower()


def test_agent_rejects_invalid_tool_argument_type():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": "fifteen", "b": 30},
        ),
        LLMResponse(
            content="I could not calculate the result.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 times 30.")

    assert isinstance(result, LLMResponse)
    assert result.content == "I could not calculate the result."

    assert fake_llm.chat.call_count == 2

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert "type" in tool_message.content.lower()


def test_agent_rejects_invalid_tool_argument_type():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": "fifteen", "b": 30},
        ),
        LLMResponse(
            content="I could not calculate the result.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 times 30.")

    assert isinstance(result, LLMResponse)
    assert result.content == "I could not calculate the result."

    assert fake_llm.chat.call_count == 2

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert "type" in tool_message.content.lower()


def test_agent_rejects_invalid_tool_argument_type():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": "fifteen", "b": 30},
        ),
        LLMResponse(
            content="I could not calculate the result.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 times 30.")

    assert isinstance(result, LLMResponse)
    assert result.content == "I could not calculate the result."

    assert fake_llm.chat.call_count == 2

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert "type" in tool_message.content.lower()


def test_agent_supports_default_tool_arguments():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="calculator",
            arguments={"a": 15},
        ),
        LLMResponse(
            content="The result is 150.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def calculator(a: int, b: int = 10) -> int:
        """Multiply two numbers."""
        return a * b

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
    )

    result = agent.run("Calculate 15 times the default value.")

    assert isinstance(result, LLMResponse)
    assert result.content == "The result is 150."

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert tool_message.content == "150"


def test_agent_supports_optional_tool_arguments():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="search",
            arguments={
                "query": "AI agents",
                "limit": None,
            },
        ),
        LLMResponse(
            content="Search completed.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def search(query: str, limit: int | None = None) -> str:
        """Search for a query."""
        return f"Searching for {query} with limit {limit}"

    agent = Agent(
        llm=fake_llm,
        tools={"search": search},
    )

    result = agent.run("Search for AI agents.")

    assert isinstance(result, LLMResponse)
    assert result.content == "Search completed."

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert tool_message.content == "Searching for AI agents with limit None"


def test_agent_tool_schema_matches_runtime_optional_type():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="search",
            arguments={
                "query": "AI agents",
                "limit": 10,
            },
        ),
        LLMResponse(
            content="Search completed.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def search(query: str, limit: int | None = None) -> str:
        """Search for a query."""
        return f"{query}:{limit}"

    agent = Agent(
        llm=fake_llm,
        tools={"search": search},
    )

    result = agent.run("Search for AI agents.")

    assert isinstance(result, LLMResponse)
    assert result.content == "Search completed."

    schema = agent.tool_schemas()["search"]

    assert schema["parameters"]["properties"]["query"] == {
        "type": "string",
    }

    assert schema["parameters"]["properties"]["limit"] == {
        "type": "integer",
    }

    assert "limit" not in schema["parameters"]["required"]

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert tool_message.content == "AI agents:10"


def test_agent_supports_list_tool_arguments():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="search",
            arguments={
                "queries": ["AI agents", "LLM tools"],
            },
        ),
        LLMResponse(
            content="Search completed.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def search(queries: list[str]) -> str:
        """Search multiple queries."""
        return ", ".join(queries)

    agent = Agent(
        llm=fake_llm,
        tools={"search": search},
    )

    result = agent.run("Search for AI agents and LLM tools.")

    assert isinstance(result, LLMResponse)
    assert result.content == "Search completed."

    schema = agent.tool_schemas()["search"]

    assert schema["parameters"]["properties"]["queries"] == {
        "type": "array",
        "items": {
            "type": "string",
        },
    }

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert tool_message.content == "AI agents, LLM tools"


def test_agent_supports_dict_tool_arguments():
    fake_llm = Mock()

    fake_llm.chat.side_effect = [
        ToolCall(
            tool="search",
            arguments={
                "filters": {
                    "language": "python",
                    "topic": "agents",
                },
            },
        ),
        LLMResponse(
            content="Search completed.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
    ]

    def search(filters: dict[str, str]) -> str:
        """Search with filters."""
        return f"{filters['language']}:{filters['topic']}"

    agent = Agent(
        llm=fake_llm,
        tools={"search": search},
    )

    result = agent.run("Search for Python agents.")

    assert isinstance(result, LLMResponse)
    assert result.content == "Search completed."

    schema = agent.tool_schemas()["search"]

    assert schema["parameters"]["properties"]["filters"] == {
        "type": "object",
        "additionalProperties": {
            "type": "string",
        },
    }

    tool_message = agent.messages[-2]

    assert tool_message.role == "tool"
    assert tool_message.content == "python:agents"
