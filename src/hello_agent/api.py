from fastapi import Depends, FastAPI
from pydantic import BaseModel

from hello_agent.agent import Agent
from hello_agent.config import LLMConfig
from hello_agent.llm import LLMClient
from hello_agent.types import LLMResponse

app = FastAPI(title="hello-agent")


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    content: str


def get_agent() -> Agent:
    return Agent(llm=LLMClient(LLMConfig()))


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    agent: Agent = Depends(get_agent),  # noqa: B008 — this is the standard FastAPI DI pattern
) -> ChatResponse:
    result = agent.run(request.prompt)

    content = result.content if isinstance(result, LLMResponse) else str(result)

    return ChatResponse(content=content)
