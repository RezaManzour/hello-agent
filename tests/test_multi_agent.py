from unittest.mock import Mock

from hello_agent.agent import Agent
from hello_agent.multi_agent import AgentTool
from hello_agent.types import LLMResponse, ToolCall


def test_agent_tool_runs_wrapped_agent_and_returns_content():
    specialist_llm = Mock()
    specialist_llm.chat.return_value = LLMResponse(
        content="The capital of France is Paris.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
    )

    specialist_agent = Agent(llm=specialist_llm)

    tool = AgentTool(
        agent=specialist_agent,
        description="Ask the geography specialist a question.",
    )

    result = tool(query="What is the capital of France?")

    assert result == "The capital of France is Paris."


def test_orchestrator_agent_delegates_to_specialist_agent():
    specialist_llm = Mock()
    specialist_llm.chat.return_value = LLMResponse(
        content="The capital of France is Paris.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
    )
    specialist_agent = Agent(llm=specialist_llm)

    specialist_tool = AgentTool(
        agent=specialist_agent,
        description="Ask the geography specialist a question.",
    )

    orchestrator_llm = Mock()
    orchestrator_llm.chat.side_effect = [
        ToolCall(
            tool="ask_geography_specialist",
            arguments={"query": "What is the capital of France?"},
        ),
        LLMResponse(
            content="According to the specialist, it's Paris.",
            model="test-model",
            finish_reason="stop",
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        ),
    ]

    orchestrator = Agent(
        llm=orchestrator_llm,
        tools={"ask_geography_specialist": specialist_tool},
    )

    result = orchestrator.run("What is the capital of France?")

    assert isinstance(result, LLMResponse)
    assert result.content == "According to the specialist, it's Paris."

    tool_message = orchestrator.messages[-2]
    assert tool_message.role == "tool"
    assert tool_message.content == "The capital of France is Paris."
