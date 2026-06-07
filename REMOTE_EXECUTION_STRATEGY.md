# Remote Execution Strategy: Managing Fleets from the Jetson Nano

To make the Jetson Nano the central "command node" that manages both its own local disk and a fleet of remote machines (via SSH), we need to expand the Agent Harness.

The goal is to allow the local Qwen 3B agent to SSH into remote machines for basic health checks and tests, while allowing the Cursor SDK (for complex tasks) to execute heavy code changes on those same remote machines.

Here is the architectural approach for seamless remote and local execution.

## 1. Expanding the Toolchain for Qwen 3B (`tools.py`)

For simple tasks (e.g., "Is the web server running on the remote database node?"), we don't need the Cursor SDK. We can give Qwen 3B direct access to an `asyncssh` toolset so it can hop into remote machines, run bash commands, and read the results natively.

First, add `asyncssh` to your requirements:
```bash
pip install asyncssh
```

Then, add an SSH execution tool to `tools.py`:

```python
import asyncssh
import asyncio

async def ssh_execute_command(host: str, username: str, command: str, key_path: str = "~/.ssh/id_rsa") -> str:
    """
    Executes a shell command on a remote machine via SSH and returns the output.
    This allows the local Qwen agent to run basic tests remotely.
    """
    try:
        async with asyncssh.connect(host, username=username, client_keys=[key_path]) as conn:
            result = await conn.run(command, check=False)
            if result.exit_status == 0:
                return f"[Success]\n{result.stdout}"
            else:
                return f"[Error: Exit Code {result.exit_status}]\n{result.stderr}"
    except Exception as e:
        return f"[SSH Connection Failed] {str(e)}"

# Add to toolchain mapping
TOOLCHAIN = {
    "web_search": web_search,
    "read_local_logs": read_local_logs,
    "ssh_execute_command": ssh_execute_command
}
```

Now, if a prompt asks to "Check if nginx is running on 192.168.1.50", Qwen 3B will output a tool call for `ssh_execute_command`, securely retrieve the systemctl status, and summarize it for you. **Zero cloud compute used.**

## 2. Remote File Editing with the Cursor SDK (The `sshfs` Method)

When a task is `COMPLEX` (e.g., "Refactor the authentication middleware on the remote staging server"), the local Qwen router offloads the request to the **Cursor SDK**. 

The Cursor SDK's `LocalAgentOptions(cwd=...)` is designed to manipulate files on a mounted filesystem. To allow the Cursor SDK to seamlessly edit remote files, we can use **SSHFS (SSH Filesystem)** on the Jetson Nano.

This mounts the remote machine's disk directly onto the Jetson. The Cursor SDK cloud agent interacts with it as if it were a local directory, unaware that the files physically reside on a remote node.

### Setup SSHFS on the Jetson
```bash
sudo apt update && sudo apt install sshfs
mkdir -p /home/ameyades/remote_mounts/staging_server

# Mount the remote server's codebase onto the Jetson
sshfs user@192.168.1.50:/var/www/html /home/ameyades/remote_mounts/staging_server
```

### Dynamic SDK Routing in `router.py`
When dispatching to the Cursor SDK, we just change the `cwd` (Current Working Directory) to point to the SSHFS mount:

```python
import os
from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

async def _call_cursor_sdk(self, prompt: str, target_machine: str = "local") -> str:
    """
    Routes the complex execution to the Cursor SDK.
    `target_machine` determines which disk the SDK manipulates.
    """
    
    # Map environments to physical or SSHFS-mounted paths on the Jetson
    workspaces = {
        "local": os.getcwd(),
        "staging": "/home/ameyades/remote_mounts/staging_server",
        "production": "/home/ameyades/remote_mounts/prod_server"
    }
    
    target_cwd = workspaces.get(target_machine, os.getcwd())
    print(f"[Router] Dispatching to Cursor SDK targeting path: {target_cwd}")
    
    loop = asyncio.get_event_loop()
    
    def run_cursor_agent():
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=self.cursor_api_key,
                model="composer-2.5",
                # The cloud agent natively edits the remote files via the SSHFS mount!
                local=LocalAgentOptions(cwd=target_cwd),
            )
        )
        return result

    result = await loop.run_in_executor(None, run_cursor_agent)
    return result.result
```

## The Workflow in Action

1. **User asks:** *"Check the database logs on the staging server."*
   - **Triage Router:** Simple task.
   - **Execution:** Local Qwen 3B triggers the `ssh_execute_command` tool, runs `tail -n 50 /var/log/postgresql`, and summarizes it natively on the Jetson.
   
2. **User asks:** *"The logs show a race condition in `db_pool.js`. Rewrite the connection logic on the staging server."*
   - **Triage Router:** Complex task.
   - **Execution:** Router calls `_call_cursor_sdk(prompt, target_machine="staging")`. The Cursor Cloud Agent spins up, reads `db_pool.js` over the SSHFS mount, writes the refactored code directly to the remote server, and closes the connection.

This creates a master node (the Jetson Nano) that maximizes the value of your $20/month Cursor subscription by administering, testing, and completely rewriting codebases across an entire fleet of remote servers.