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


def test_mcp_client_calls_add_tool_through_protocol():
    from mcp import Client

    server = create_server()

    async def run():
        async with Client(server) as client:
            result = await client.call_tool("add", {"a": 10, "b": 5})
            return result

    result = asyncio.run(run())

    assert result.structured_content == {"result": 15}


def test_mcp_client_lists_tools_through_protocol():
    from mcp import Client

    server = create_server()

    async def run():
        async with Client(server) as client:
            tools = await client.list_tools()
            return tools

    result = asyncio.run(run())
    tool_names = [tool.name for tool in result.tools]

    assert "add" in tool_names


def test_mcp_tool_bridge_calls_server_synchronously():
    from hello_agent.mcp_bridge import MCPToolBridge

    server = create_server()

    bridge = MCPToolBridge(server=server, tool_name="add")

    result = bridge(a=3, b=4)

    assert result == {"result": 7}
