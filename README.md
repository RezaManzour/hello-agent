# hello-agent

A from-scratch, test-driven implementation of an LLM agent framework — built as a hands-on path into agentic AI engineering rather than as a wrapper around an existing framework.

No LangChain, no CrewAI, no AutoGPT. Every piece — the LLM client, the tool-calling loop, memory, retrieval, MCP integration, and multi-agent delegation — is implemented and tested from first principles in Python.

## Why this project exists

Most "agent" tutorials show you how to call a framework's API. This project instead answers: *what actually happens under the hood when an agent calls a tool, remembers a conversation, retrieves a document, or delegates to another agent?* Every feature below was built with strict TDD (red → green → refactor), one small, verified step at a time.

## What it can do

- **Talk to an LLM** through a clean, provider-agnostic `Message` protocol (currently backed by Hugging Face's `InferenceClient`)
- **Call tools** — Python functions are inspected automatically to generate JSON-schema tool definitions, with full argument validation (types, required/optional, lists, dicts, unions, defaults)
- **Handle tool results and errors** as first-class, structured data (`ToolResult`), not string-parsing hacks
- **Remember conversations** — save and reload full sessions to/from JSON, with automatic context-window trimming once a conversation gets too long
- **Retrieve relevant context (RAG)** — embed documents, store them in an in-memory vector store, and let the agent search them via cosine similarity, exposed as an ordinary tool
- **Speak MCP** — run a real [Model Context Protocol](https://modelcontextprotocol.io/) server, connect to it with the official client over the real protocol, and bridge any MCP tool into the agent's tool-calling interface
- **Delegate between agents** — wrap one agent as a callable tool for another, enabling simple orchestrator → specialist multi-agent patterns

## Architecture

```
User
 ↓
Message(role="user")
 ↓
Agent.run()  ──────────────────────────────┐
 ↓                                         │ (loop, up to max_iterations)
LLMClient.chat()                           │
 ↓                                         │
ToolCall?  ──yes──▶ Agent._run_tool()      │
 │                    ↓                    │
 │                 ToolResult               │
 │                    ↓                    │
 │            ToolResult.to_message()      │
 │                    ↓                    │
 │            Message(role="tool") ─────────┘
 │
 no
 ↓
LLMResponse (final answer)
```

Tools are just `Callable[..., object]` — a plain function, or any object with a type-annotated `__call__`. This is what lets a RAG retriever, an MCP-bridged tool, and a whole other `Agent` all plug into the exact same tool-calling loop with zero special-casing.

## Project structure

```
src/hello_agent/
├── agent.py         # Agent: the tool-calling loop, sessions, context limits
├── llm.py           # LLMClient: chat, streaming, structured outputs
├── config.py        # LLMConfig, EmbeddingConfig
├── types.py         # Message, ToolCall, ToolResult, LLMResponse
├── exceptions.py     # LLM error hierarchy
├── retrieval.py      # VectorStore, Retriever (RAG)
├── embedding.py       # EmbeddingClient
├── mcp_server.py      # A real MCP server exposing sample tools
├── mcp_bridge.py       # Sync adapter: MCP tool → Agent tool
└── multi_agent.py       # AgentTool: use one Agent as another's tool

tests/                    # One test file per module, TDD throughout
```

## Getting started

```bash
git clone https://github.com/RezaManzour/hello-agent.git
cd hello-agent
uv sync
cp .env.example .env   # add your HF_TOKEN
uv run pytest -q
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Example

```python
from hello_agent.agent import Agent
from hello_agent.llm import LLMClient
from hello_agent.config import LLMConfig

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."

agent = Agent(
    llm=LLMClient(LLMConfig()),
    tools={"get_weather": get_weather},
)

result = agent.run("What's the weather like in Tehran?")
print(result.content)
```

Swap `get_weather` for a `Retriever`, an `MCPToolBridge`, or an `AgentTool` wrapping another agent — the calling code doesn't change.

## Development

This project is built with strict test-driven development:

```bash
uv run pytest -q          # run the full suite
uv run ruff check .       # lint
uv run ruff format .      # format
```

Every feature was added red → green → refactor, one commit at a time — the [commit history](https://github.com/RezaManzour/hello-agent/commits/main/) is effectively a build log of the whole project.

## Roadmap

| Phase | Status |
|---|---|
| 0 — Environment & tooling | ✅ |
| 1 — LLM engineering | ✅ |
| 2 — Tool calling & agents | ✅ |
| 3 — Memory (persistence + context limits) | ✅ |
| 4 — RAG | ✅ |
| 5 — MCP | ✅ |
| 6 — Multi-agent systems | ✅ |
| 7 — Evaluation | 🔜 |
| 8 — AI security | ⬜ |
| 9 — Deployment & observability | ⬜ |
| 10 — Production agent systems | ⬜ |

## License

MIT
