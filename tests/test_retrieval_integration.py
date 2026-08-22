from unittest.mock import Mock, patch

import numpy as np

from hello_agent.config import EmbeddingConfig
from hello_agent.embedding import EmbeddingClient
from hello_agent.retrieval import Retriever, VectorStore


def test_embed_and_store_then_retrieve_most_similar():
    with patch("hello_agent.embedding.InferenceClient") as mock_cls:
        mock_instance = Mock()

        def fake_embed(text, model):
            vectors = {
                "The cat sat on the mat.": np.array([1.0, 0.0, 0.0]),
                "Dogs are loyal animals.": np.array([0.0, 1.0, 0.0]),
                "Cats are independent pets.": np.array([0.9, 0.1, 0.0]),
                "Tell me about cats": np.array([1.0, 0.0, 0.0]),
            }
            return vectors[text]

        mock_instance.feature_extraction.side_effect = fake_embed
        mock_cls.return_value = mock_instance

        embedding_client = EmbeddingClient(EmbeddingConfig())
        store = VectorStore()

        documents = [
            "The cat sat on the mat.",
            "Dogs are loyal animals.",
            "Cats are independent pets.",
        ]

        for doc in documents:
            vector = embedding_client.embed(doc)
            store.add(text=doc, vector=vector)

        query_vector = embedding_client.embed("Tell me about cats")
        results = store.search(query_vector=query_vector, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "The cat sat on the mat."
        assert results[1][0] == "Cats are independent pets."


def test_retriever_tool_returns_joined_top_results():
    from hello_agent.retrieval import Retriever

    with patch("hello_agent.embedding.InferenceClient") as mock_cls:
        mock_instance = Mock()

        def fake_embed(text, model):
            vectors = {
                "The cat sat on the mat.": np.array([1.0, 0.0, 0.0]),
                "Dogs are loyal animals.": np.array([0.0, 1.0, 0.0]),
                "Cats are independent pets.": np.array([0.9, 0.1, 0.0]),
                "Tell me about cats": np.array([1.0, 0.0, 0.0]),
            }
            return vectors[text]

        mock_instance.feature_extraction.side_effect = fake_embed
        mock_cls.return_value = mock_instance

        embedding_client = EmbeddingClient(EmbeddingConfig())
        store = VectorStore()

        for doc in [
            "The cat sat on the mat.",
            "Dogs are loyal animals.",
            "Cats are independent pets.",
        ]:
            store.add(text=doc, vector=embedding_client.embed(doc))

        retriever = Retriever(
            embedding_client=embedding_client,
            store=store,
            top_k=2,
        )

        result = retriever(query="Tell me about cats")

        assert isinstance(result, str)
        assert "The cat sat on the mat." in result
        assert "Cats are independent pets." in result
        assert "Dogs are loyal animals." not in result


def test_agent_uses_retriever_as_tool():
    from unittest.mock import Mock, patch

    from hello_agent.agent import Agent
    from hello_agent.types import LLMResponse, ToolCall

    with patch("hello_agent.embedding.InferenceClient") as mock_cls:
        mock_instance = Mock()

        def fake_embed(text, model):
            vectors = {
                "The cat sat on the mat.": np.array([1.0, 0.0, 0.0]),
                "Dogs are loyal animals.": np.array([0.0, 1.0, 0.0]),
                "Cats are independent pets.": np.array([0.9, 0.1, 0.0]),
                "Tell me about cats": np.array([1.0, 0.0, 0.0]),
            }
            return vectors[text]

        mock_instance.feature_extraction.side_effect = fake_embed
        mock_cls.return_value = mock_instance

        embedding_client = EmbeddingClient(EmbeddingConfig())
        store = VectorStore()

        for doc in [
            "The cat sat on the mat.",
            "Dogs are loyal animals.",
            "Cats are independent pets.",
        ]:
            store.add(text=doc, vector=embedding_client.embed(doc))

        retriever = Retriever(
            embedding_client=embedding_client,
            store=store,
            top_k=2,
        )

        fake_llm = Mock()
        fake_llm.chat.side_effect = [
            ToolCall(
                tool="retrieve",
                arguments={"query": "Tell me about cats"},
            ),
            LLMResponse(
                content="Cats are commonly kept as pets.",
                model="test-model",
                finish_reason="stop",
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28,
            ),
        ]

        agent = Agent(
            llm=fake_llm,
            tools={"retrieve": retriever},
        )

        result = agent.run("Tell me about cats.")

        assert isinstance(result, LLMResponse)
        assert result.content == "Cats are commonly kept as pets."

        tool_message = agent.messages[-2]
        assert tool_message.role == "tool"
        assert "The cat sat on the mat." in tool_message.content
        assert "Cats are independent pets." in tool_message.content
