import asyncio

from hello_agent.mcp_server import create_server


def test_mcp_server_exposes_add_tool():
    server = create_server()

    async def run():
        tools = await server.list_tools()
        return tools

    tools = asyncio.run(run())
    tool_names = [tool.name for tool in tools]

    assert "add" in tool_names


def test_mcp_server_add_tool_returns_correct_result():
    server = create_server()

    async def run():
        return await server.call_tool("add", {"a": 3, "b": 4})

    result = asyncio.run(run())

    assert result.structured_content == {"result": 7}
