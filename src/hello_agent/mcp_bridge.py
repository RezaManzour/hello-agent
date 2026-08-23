import asyncio

from mcp import Client
from mcp.server import MCPServer


class MCPToolBridge:
    def __init__(self, server: MCPServer, tool_name: str):
        self.server = server
        self.tool_name = tool_name

    def __call__(self, **kwargs: object) -> object:
        async def run() -> object:
            async with Client(self.server) as client:
                result = await client.call_tool(self.tool_name, kwargs)
                return result.structured_content

        return asyncio.run(run())
