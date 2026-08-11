from hello_agent.config import LLMConfig
from hello_agent.llm import LLMClient
from hello_agent.types import Message


def main():
    config = LLMConfig()
    client = LLMClient(config)

    response = client.chat(
        [
            Message(
                role="user",
                content="Explain what an AI agent is in one sentence.",
            )
        ]
    )

    print(response.content)
    print(f"Model: {response.model}")
    print(f"Tokens: {response.total_tokens}")


if __name__ == "__main__":
    main()
