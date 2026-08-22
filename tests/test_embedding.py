from unittest.mock import Mock, patch

import numpy as np

from hello_agent.config import EmbeddingConfig
from hello_agent.embedding import EmbeddingClient


def test_embedding_client_returns_list_of_floats():
    with patch("hello_agent.embedding.InferenceClient") as mock_cls:
        mock_instance = Mock()
        mock_instance.feature_extraction.return_value = np.array([0.1, 0.2, 0.3])
        mock_cls.return_value = mock_instance

        config = EmbeddingConfig()
        client = EmbeddingClient(config)

        result = client.embed("Hello world")

        assert result == [0.1, 0.2, 0.3]
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


def test_embedding_client_calls_feature_extraction_with_correct_args():
    with patch("hello_agent.embedding.InferenceClient") as mock_cls:
        mock_instance = Mock()
        mock_instance.feature_extraction.return_value = np.array([0.5])
        mock_cls.return_value = mock_instance

        config = EmbeddingConfig()
        client = EmbeddingClient(config)

        client.embed("test text")

        mock_instance.feature_extraction.assert_called_once_with(
            "test text",
            model=config.model,
        )
