from unittest.mock import Mock

from hello_agent.agent import Agent
from hello_agent.types import LLMResponse, ToolCall


def test_agent_emits_run_start_and_run_end_events():
    fake_llm = Mock()
    fake_llm.chat.return_value = LLMResponse(
        content="Paris.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    events = []

    def on_event(name, data):
        events.append((name, data))

    agent = Agent(llm=fake_llm, on_event=on_event)

    agent.run("What is the capital of France?")

    event_names = [name for name, _ in events]
    assert "run_start" in event_names
    assert "run_end" in event_names

    run_start_data = next(data for name, data in events if name == "run_start")
    assert run_start_data["prompt"] == "What is the capital of France?"

    run_end_data = next(data for name, data in events if name == "run_end")
    assert run_end_data["content"] == "Paris."


def test_agent_emits_tool_call_and_tool_result_events():
    fake_llm = Mock()
    fake_llm.chat.side_effect = [
        ToolCall(tool="calculator", arguments={"a": 2, "b": 3}),
        LLMResponse(
            content="5.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        return a + b

    events = []

    def on_event(name, data):
        events.append((name, data))

    agent = Agent(
        llm=fake_llm,
        tools={"calculator": calculator},
        on_event=on_event,
    )

    agent.run("What is 2 + 3?")

    event_names = [name for name, _ in events]
    assert "tool_call" in event_names
    assert "tool_result" in event_names

    tool_call_data = next(data for name, data in events if name == "tool_call")
    assert tool_call_data["tool"] == "calculator"
    assert tool_call_data["arguments"] == {"a": 2, "b": 3}

    tool_result_data = next(data for name, data in events if name == "tool_result")
    assert tool_result_data["tool"] == "calculator"
    assert tool_result_data["is_error"] is False


def test_agent_without_on_event_still_works():
    fake_llm = Mock()
    fake_llm.chat.return_value = LLMResponse(
        content="Paris.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    agent = Agent(llm=fake_llm)

    result = agent.run("What is the capital of France?")

    assert isinstance(result, LLMResponse)
    assert result.content == "Paris."
