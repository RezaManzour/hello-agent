from hello_agent.agent import Agent
from hello_agent.types import LLMResponse


class AgentTool:
    def __init__(self, agent: Agent, description: str = ""):
        self.agent = agent
        self.description = description

    def __call__(self, query: str) -> str:
        result = self.agent.run(query)

        if isinstance(result, LLMResponse):
            return result.content

        return str(result)
