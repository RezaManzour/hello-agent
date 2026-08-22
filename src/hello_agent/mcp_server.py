from mcp.server import MCPServer


def create_server() -> MCPServer:
    server = MCPServer(name="hello-agent-mcp")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return server
