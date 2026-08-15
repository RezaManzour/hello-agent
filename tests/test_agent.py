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
