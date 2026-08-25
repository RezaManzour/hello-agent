from unittest.mock import Mock

from hello_agent.agent import Agent
from hello_agent.evaluation import EvalCase, run_eval_suite
from hello_agent.types import LLMResponse, ToolCall


def test_eval_case_passes_when_content_matches():
    fake_llm = Mock()
    fake_llm.chat.return_value = LLMResponse(
        content="The capital of France is Paris.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
    )

    def agent_factory():
        return Agent(llm=fake_llm)

    cases = [
        EvalCase(
            prompt="What is the capital of France?",
            expected_content_contains="Paris",
        ),
    ]

    report = run_eval_suite(agent_factory, cases)

    assert report.total == 1
    assert report.passed == 1
    assert report.failed == 0
    assert report.results[0].passed is True


def test_eval_case_fails_when_content_does_not_match():
    fake_llm = Mock()
    fake_llm.chat.return_value = LLMResponse(
        content="I'm not sure.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
    )

    def agent_factory():
        return Agent(llm=fake_llm)

    cases = [
        EvalCase(
            prompt="What is the capital of France?",
            expected_content_contains="Paris",
        ),
    ]

    report = run_eval_suite(agent_factory, cases)

    assert report.total == 1
    assert report.passed == 0
    assert report.failed == 1
    assert report.results[0].passed is False


def test_eval_case_checks_expected_tool_was_called():
    fake_llm = Mock()
    fake_llm.chat.side_effect = [
        ToolCall(tool="calculator", arguments={"a": 2, "b": 3}),
        LLMResponse(
            content="The result is 5.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
        ),
    ]

    def calculator(a: int, b: int) -> int:
        return a + b

    def agent_factory():
        return Agent(llm=fake_llm, tools={"calculator": calculator})

    cases = [
        EvalCase(
            prompt="What is 2 + 3?",
            expected_tool="calculator",
            expected_content_contains="5",
        ),
    ]

    report = run_eval_suite(agent_factory, cases)

    assert report.passed == 1
    assert report.results[0].passed is True


def test_eval_case_fails_when_expected_tool_not_called():
    fake_llm = Mock()
    fake_llm.chat.return_value = LLMResponse(
        content="The result is 5.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
    )

    def calculator(a: int, b: int) -> int:
        return a + b

    def agent_factory():
        return Agent(llm=fake_llm, tools={"calculator": calculator})

    cases = [
        EvalCase(
            prompt="What is 2 + 3?",
            expected_tool="calculator",
        ),
    ]

    report = run_eval_suite(agent_factory, cases)

    assert report.passed == 0
    assert report.failed == 1
