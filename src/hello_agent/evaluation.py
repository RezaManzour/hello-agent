import json
from collections.abc import Callable
from dataclasses import dataclass, field

from hello_agent.agent import Agent
from hello_agent.types import LLMResponse


@dataclass
class EvalCase:
    prompt: str
    expected_tool: str | None = None
    expected_content_contains: str | None = None


@dataclass
class EvalResult:
    case: EvalCase
    passed: bool
    reason: str = ""


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


def _was_tool_called(agent: Agent, tool_name: str) -> bool:
    for message in agent.messages:
        if message.role != "assistant":
            continue

        try:
            data = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(data, dict) and data.get("tool") == tool_name:
            return True

    return False


def _evaluate_case(agent_factory: Callable[[], Agent], case: EvalCase) -> EvalResult:
    agent = agent_factory()
    result = agent.run(case.prompt)

    if case.expected_tool is not None and not _was_tool_called(
        agent, case.expected_tool
    ):
        return EvalResult(
            case=case,
            passed=False,
            reason=f"Expected tool '{case.expected_tool}' was not called.",
        )

    if case.expected_content_contains is not None:
        content = result.content if isinstance(result, LLMResponse) else str(result)

        if case.expected_content_contains not in content:
            return EvalResult(
                case=case,
                passed=False,
                reason=(
                    f"Expected content to contain "
                    f"'{case.expected_content_contains}', got: {content!r}"
                ),
            )

    return EvalResult(case=case, passed=True)


def run_eval_suite(
    agent_factory: Callable[[], Agent],
    cases: list[EvalCase],
) -> EvalReport:
    report = EvalReport()

    for case in cases:
        report.results.append(_evaluate_case(agent_factory, case))

    return report
