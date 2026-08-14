from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from huggingface_hub.errors import HfHubHTTPError

from hello_agent.config import LLMConfig
from hello_agent.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
)
from hello_agent.llm import LLMClient
from hello_agent.types import AgentDefinition, Message


def make_hf_error(status_code: int) -> HfHubHTTPError:
    request = httpx.Request(
        "POST",
        "https://router.huggingface.co/v1/chat/completions",
    )

    response = httpx.Response(
        status_code,
        request=request,
    )

    return HfHubHTTPError(
        f"HTTP {status_code}",
        response=response,
    )


def test_llm_client_returns_response():
    fake_response = SimpleNamespace(
        model="test-model",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="Hello from the mock model."
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=6,
            total_tokens=16,
        ),
    )

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        response = client.chat(
            [
                Message(
                    role="user",
                    content="Say hello.",
                )
            ]
        )

    assert response.content == "Hello from the mock model."
    assert response.model == "test-model"
    assert response.total_tokens == 16


def test_llm_client_handles_authentication_error():
    fake_client = Mock()
    fake_client.chat.completions.create.side_effect = make_hf_error(401)

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        try:
            client.chat(
                [
                    Message(
                        role="user",
                        content="Hello",
                    )
                ]
            )
        except LLMAuthenticationError as exc:
            assert "authentication" in str(exc).lower()
        else:
            raise AssertionError(
                "LLMAuthenticationError was not raised"
            )


def test_llm_client_handles_rate_limit_error():
    fake_client = Mock()
    fake_client.chat.completions.create.side_effect = make_hf_error(429)

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        try:
            client.chat(
                [
                    Message(
                        role="user",
                        content="Hello",
                    )
                ]
            )
        except LLMRateLimitError as exc:
            assert "rate limit" in str(exc).lower()
        else:
            raise AssertionError(
                "LLMRateLimitError was not raised"
            )


def test_llm_client_handles_provider_error():
    fake_client = Mock()
    fake_client.chat.completions.create.side_effect = make_hf_error(500)

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        try:
            client.chat(
                [
                    Message(
                        role="user",
                        content="Hello"
                    )
                ]
            )
        except LLMProviderError as exc:
            assert "provider error" in str(exc).lower()
            assert "500" in str(exc)
        else:
            raise AssertionError(
                "LLMProviderError was not raised"
            )


def test_llm_client_passes_generation_parameters():
    fake_response = SimpleNamespace(
        model="test-model",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="Test response."
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
        ),
    )

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        client.chat(
            [
                Message(
                    role="user",
                    content="Test parameters.",
                )
            ]
        )

    fake_client.chat.completions.create.assert_called_once()

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs

    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 512


def test_llm_client_returns_structured_output():
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"definition": "An AI agent acts toward a goal.", "confidence": 0.95}'
                ),
                finish_reason="stop",
            )
        ],
        model="test-model",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        result = client.chat(
            [
                Message(
                    role="user",
                    content="Explain what an AI agent is.",
                )
            ],
            response_model=AgentDefinition,
        )

    assert isinstance(result, AgentDefinition)
    assert result.definition == "An AI agent acts toward a goal."
    assert result.confidence == 0.95


def test_llm_client_sends_structured_output_schema():
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"definition": "An AI agent acts toward a goal.", "confidence": 0.95}'
                ),
                finish_reason="stop",
            )
        ],
        model="test-model",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        client.chat(
            [
                Message(
                    role="user",
                    content="Explain what an AI agent is.",
                )
            ],
            response_model=AgentDefinition,
        )

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    response_format = call_kwargs["response_format"]

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "AgentDefinition"
    assert response_format["json_schema"]["schema"] == AgentDefinition.model_json_schema()


def test_llm_client_handles_invalid_structured_output():
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"definition": "Missing confidence"}'
                ),
                finish_reason="stop",
            )
        ],
        model="test-model",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        try:
            client.chat(
                [
                    Message(
                        role="user",
                        content="Explain what an AI agent is.",
                    )
                ],
                response_model=AgentDefinition,
            )
            raise AssertionError("Expected LLMProviderError")
        except Exception as exc:
            assert "structured" in str(exc).lower()


def test_llm_client_streams_text():
    fake_chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello ")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="world!")
                )
            ]
        ),
    ]

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = iter(fake_chunks)

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        result = list(
            client.chat_stream(
                [
                    Message(
                        role="user",
                        content="Say hello.",
                    )
                ]
            )
        )

    assert result == ["Hello ", "world!"]


def test_llm_client_stream_ignores_empty_chunks():
    fake_chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None)
                )
            ]
        ),
    ]

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = iter(fake_chunks)

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        result = list(
            client.chat_stream(
                [
                    Message(
                        role="user",
                        content="Say hello.",
                    )
                ]
            )
        )

    assert result == ["Hello"]


def test_llm_client_stream_passes_parameters():
    fake_chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Test")
                )
            ]
        )
    ]

    fake_client = Mock()
    fake_client.chat.completions.create.return_value = iter(fake_chunks)

    with patch(
        "hello_agent.llm.InferenceClient",
        return_value=fake_client,
    ):
        client = LLMClient(LLMConfig())

        list(
            client.chat_stream(
                [
                    Message(
                        role="user",
                        content="Test parameters.",
                    )
                ]
            )
        )

    fake_client.chat.completions.create.assert_called_once()

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs

    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 512
    assert call_kwargs["stream"] is True
