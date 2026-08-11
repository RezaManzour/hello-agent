from hello_agent.config import LLMConfig


def test_default_config():
    config = LLMConfig()

    assert config.model
    assert config.temperature == 0.7
    assert config.max_tokens == 512
    assert config.timeout == 30
