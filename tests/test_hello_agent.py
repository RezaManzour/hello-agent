from hello_agent import main as main_module
from hello_agent.types import LLMResponse


class FakeLLMClient:
    def __init__(self, config) -> None:
        self.config = config
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return LLMResponse(
            content="An AI agent is a program that uses a model to choose actions.",
            model="fake-model",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=12,
            total_tokens=22,
        )


def test_main_prints_answer_model_and_tokens(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "LLMClient", FakeLLMClient)

    main_module.main()

    captured = capsys.readouterr()
    assert captured.out == (
        "An AI agent is a program that uses a model to choose actions.\n"
        "Model: fake-model\n"
        "Tokens: 22\n"
    )
