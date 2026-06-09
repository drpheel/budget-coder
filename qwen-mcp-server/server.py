import asyncio
import json
import os
import subprocess
from typing import Optional, List, Dict, Any

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from openai import AsyncOpenAI

server = Server("qwen-mcp-server")

# --- Qwen 3B Sub-Agent Orchestrator ---

openai_client = AsyncOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

QWEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "internal_read_file",
            "description": "Reads the contents of a file to inspect code or config.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "internal_run_command",
            "description": "Runs a diagnostic shell command (e.g., pip check, ls, cat).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"]
            }
        }
    }
]

def execute_internal_tool(name: str, arguments: dict) -> str:
    if name == "internal_read_file":
        path = arguments.get("path")
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
            
    elif name == "internal_run_command":
        command = arguments.get("command")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=os.path.expanduser("~")
            )
            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"Stdout:\n{result.stdout}\n"
            if result.stderr:
                output += f"Stderr:\n{result.stderr}\n"
            return output
        except Exception as e:
            return f"Error running command: {str(e)}"
            
    return f"Unknown tool: {name}"

def get_qwen_system_prompt(base_prompt: str) -> str:
    return f"""{base_prompt}

You have access to the following tools:
1. internal_read_file(path: str): Reads the contents of a file to inspect code or config.
2. internal_run_command(command: str): Runs a diagnostic shell command (e.g., pip check, ls, cat).

To use a tool, you MUST output a JSON block like this:
```json
{{
  "tool": "tool_name",
  "arguments": {{
    "arg_name": "arg_value"
  }}
}}
```

Wait for the tool result before continuing. If you don't need to use a tool, just output your final response."""

async def run_qwen_agent(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": get_qwen_system_prompt(system_prompt)},
        {"role": "user", "content": user_prompt}
    ]
    
    for i in range(10): 
        try:
            response = await openai_client.chat.completions.create(
                model="qwen2.5-coder-3b-instruct",
                messages=messages,
                temperature=0.2,
                max_tokens=4096
            )
        except Exception as e:
            return f"Error communicating with Qwen 3B (is llama-server running?): {str(e)}"
            
        message = response.choices[0].message
        content = message.content or ""
        print(f"--- Iteration {i} ---")
        print(f"Model output: {content}")
        
        messages.append({"role": "assistant", "content": content})
        
        # Parse manual tool call
        if "```json" in content and '"tool"' in content:
            try:
                # Extract the JSON block
                json_str = content.split("```json")[1].split("```")[0].strip()
                tool_call = json.loads(json_str)
                function_name = tool_call.get("tool")
                arguments = tool_call.get("arguments", {})
                
                print(f"Calling tool: {function_name} with args: {arguments}")
                tool_result = execute_internal_tool(function_name, arguments)
                print(f"Tool result: {tool_result}")
                
                messages.append({
                    "role": "user",
                    "content": f"Tool Result:\n{tool_result}"
                })
                continue # Loop again to let the model process the result
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Error parsing tool call: {str(e)}. Make sure to output valid JSON."
                })
                continue
                
        # If no tool call, we assume it's the final answer
        return content

    return "Agent reached maximum iterations without completing the task."

# --- MCP Tools ---

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        # Low-level tools
        types.Tool(
            name="remote_read_file",
            description="Reads specific lines from a file on the remote host.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed)", "default": 1},
                    "end_line": {"type": "integer", "description": "End line (inclusive)", "default": -1},
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="remote_write_file",
            description="Writes content to a file on the remote host.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        ),
        types.Tool(
            name="remote_run_command",
            description="Executes a shell command on the remote host and returns stdout/stderr.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"]
            }
        ),
        types.Tool(
            name="remote_list_directory",
            description="Lists files in a directory on the remote host.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the directory"}
                },
                "required": ["path"]
            }
        ),
        # High-level AI tools
        types.Tool(
            name="monitor_long_process",
            description="Spawns a background process, streams its output to Qwen 3B for continuous analysis, and returns a concise summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The long-running shell command to execute and monitor"}
                },
                "required": ["command"]
            }
        ),
        types.Tool(
            name="triage_log",
            description="Feeds a log file to Qwen 3B to identify stack traces, warnings, or anomalies, returning a filtered summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path": {"type": "string", "description": "Absolute path to the log file"}
                },
                "required": ["log_path"]
            }
        ),
        types.Tool(
            name="triage_environment",
            description="Asks Qwen 3B to investigate dependency or environment errors by checking local lockfiles and system state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "error_msg": {"type": "string", "description": "The error message to investigate"}
                },
                "required": ["error_msg"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if not arguments:
        arguments = {}

    # Low-level tools
    if name == "remote_read_file":
        path = arguments.get("path")
        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line", -1)
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            if end_line == -1:
                end_line = len(lines)
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            selected_lines = lines[start_idx:end_idx]
            content = "".join(selected_lines)
            return [types.TextContent(type="text", text=content)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error reading file: {str(e)}")]

    elif name == "remote_write_file":
        path = arguments.get("path")
        content = arguments.get("content")
        try:
            with open(path, "w") as f:
                f.write(content)
            return [types.TextContent(type="text", text=f"Successfully wrote to {path}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error writing file: {str(e)}")]

    elif name == "remote_run_command":
        command = arguments.get("command")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=os.path.expanduser("~")
            )
            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"Stdout:\n{result.stdout}\n"
            if result.stderr:
                output += f"Stderr:\n{result.stderr}\n"
            return [types.TextContent(type="text", text=output)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error running command: {str(e)}")]

    elif name == "remote_list_directory":
        path = arguments.get("path")
        try:
            items = os.listdir(path)
            details = []
            for item in items:
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    details.append(f"[DIR]  {item}")
                else:
                    details.append(f"[FILE] {item}")
            content = "\n".join(sorted(details))
            return [types.TextContent(type="text", text=content)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error listing directory: {str(e)}")]

    # High-level AI tools
    elif name == "monitor_long_process":
        command = arguments.get("command")
        try:
            # Run the command and capture output
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=os.path.expanduser("~")
            )
            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"Stdout:\n{result.stdout}\n"
            if result.stderr:
                output += f"Stderr:\n{result.stderr}\n"
                
            system_prompt = "You are a monitoring agent. Analyze the output of the following long-running process. Provide a concise summary of the exact point of failure, success, or any anomalies. Include exact file paths and line numbers if available."
            user_prompt = f"Command: {command}\n\nOutput:\n{output}"
            
            summary = await run_qwen_agent(system_prompt, user_prompt)
            return [types.TextContent(type="text", text=summary)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error running/monitoring command: {str(e)}")]

    elif name == "triage_log":
        log_path = arguments.get("log_path")
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
                
            # Truncate if too long (e.g., last 20k chars)
            if len(log_content) > 20000:
                log_content = "...[TRUNCATED]...\n" + log_content[-20000:]
                
            system_prompt = "You are a log triage agent. Analyze the provided log file content. Identify stack traces, warnings, or error patterns. Filter out noise and provide a distilled summary of anomalies. Provide exact file paths and line numbers."
            user_prompt = f"Log Path: {log_path}\n\nLog Content:\n{log_content}"
            
            summary = await run_qwen_agent(system_prompt, user_prompt)
            return [types.TextContent(type="text", text=summary)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error triaging log: {str(e)}")]

    elif name == "triage_environment":
        error_msg = arguments.get("error_msg")
        system_prompt = "You are an environment triage agent. Investigate the provided dependency or environment error. You can use your tools to inspect local lockfiles (package.json, requirements.txt) or run native check commands (pip check, npm install --dry-run). Isolate the missing dependency or version mismatch and summarize the conflict state."
        user_prompt = f"Error Message:\n{error_msg}"
        
        summary = await run_qwen_agent(system_prompt, user_prompt)
        return [types.TextContent(type="text", text=summary)]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="qwen-mcp-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
