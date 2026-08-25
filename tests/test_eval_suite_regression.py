from unittest.mock import Mock

from hello_agent.agent import Agent
from hello_agent.evaluation import EvalCase, run_eval_suite
from hello_agent.types import LLMResponse, ToolCall


def calculator(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"


def test_project_regression_suite_all_pass():
    def calculator_agent_factory():
        fake_llm = Mock()
        fake_llm.chat.side_effect = [
            ToolCall(tool="calculator", arguments={"a": 4, "b": 5}),
            LLMResponse(
                content="4 + 5 is 9.",
                model="test-model",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18,
            ),
        ]
        return Agent(llm=fake_llm, tools={"calculator": calculator})

    def greet_agent_factory():
        fake_llm = Mock()
        fake_llm.chat.side_effect = [
            ToolCall(tool="greet", arguments={"name": "Reza"}),
            LLMResponse(
                content="Hello, Reza!",
                model="test-model",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18,
            ),
        ]
        return Agent(llm=fake_llm, tools={"greet": greet})

    def plain_response_agent_factory():
        fake_llm = Mock()
        fake_llm.chat.return_value = LLMResponse(
            content="I don't need a tool for that.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
        )
        return Agent(llm=fake_llm)

    scenarios = [
        (
            calculator_agent_factory,
            EvalCase(
                prompt="What is 4 + 5?",
                expected_tool="calculator",
                expected_content_contains="9",
            ),
        ),
        (
            greet_agent_factory,
            EvalCase(
                prompt="Greet Reza.",
                expected_tool="greet",
                expected_content_contains="Reza",
            ),
        ),
        (
            plain_response_agent_factory,
            EvalCase(
                prompt="Say something without using a tool.",
                expected_content_contains="tool",
            ),
        ),
    ]

    for agent_factory, case in scenarios:
        report = run_eval_suite(agent_factory, [case])

        assert report.total == 1
        assert report.passed == 1, (
            f"Regression case failed for prompt {case.prompt!r}: "
            f"{report.results[0].reason}"
        )
