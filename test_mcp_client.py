import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python3",
        args=[os.path.abspath(os.path.join(os.path.dirname(__file__), "qwen-mcp-server/server.py"))],
    )

    print(f"Connecting to MCP server at {server_params.args[0]}...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("\n--- Available Tools ---")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
            
            print("\n--- Testing remote_list_directory ---")
            current_dir = os.path.abspath(os.path.dirname(__file__))
            print(f"Listing directory: {current_dir}")
            
            result = await session.call_tool("remote_list_directory", {"path": current_dir})
            print(result.content[0].text)
            
            print("\n--- Testing remote_read_file ---")
            target_file = os.path.join(current_dir, "run-qwen-server.sh")
            print(f"Reading file: {target_file} (lines 1-5)")
            
            result = await session.call_tool("remote_read_file", {
                "path": target_file,
                "start_line": 1,
                "end_line": 5
            })
            print(result.content[0].text)
            
            print("\n--- Testing Qwen Agent (triage_environment) ---")
            print("Asking Qwen to investigate a missing dependency...")
            # This will trigger the Qwen agent to use its internal tools to read files
            # Note: Ensure llama-server is running before executing this part
            try:
                result = await session.call_tool("triage_environment", {
                    "error_msg": "ModuleNotFoundError: No module named 'requests'"
                })
                print("\nAgent Response:")
                print(result.content[0].text)
            except Exception as e:
                print(f"Agent test failed (is llama-server running?): {e}")

if __name__ == "__main__":
    asyncio.run(main())
