from unittest.mock import Mock

from fastapi.testclient import TestClient

from hello_agent.agent import Agent
from hello_agent.api import app, get_agent
from hello_agent.types import LLMResponse


def test_chat_endpoint_returns_agent_response():
    fake_llm = Mock()
    fake_llm.chat.return_value = LLMResponse(
        content="Paris.",
        model="test-model",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    def override_get_agent():
        return Agent(llm=fake_llm)

    app.dependency_overrides[get_agent] = override_get_agent

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"prompt": "What is the capital of France?"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"content": "Paris."}


def test_chat_endpoint_requires_prompt():
    client = TestClient(app)
    response = client.post("/chat", json={})

    assert response.status_code == 422
