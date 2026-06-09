# Agent Strategy: MCP Server for Local Qwen 3B

## Objective

Rather than building a complex remote execution harness, we will expose our local Qwen2.5-Instruct-3B agent as an **MCP (Model Context Protocol) server**. This allows any compatible "software factory"—such as Cursor, Claude Code, or other AI IDEs—to seamlessly interact with and delegate tasks to the local agent.

## Core Value Proposition

The primary goal of this architecture is to **save tokens and API calls to expensive models** (like GPT-4o or Claude 3.5 Sonnet). We reserve those expensive models for complex reasoning, planning, and code generation, while offloading tedious, token-heavy, or long-running tasks to the local Qwen 3B model.

## Capabilities (Tailored for Qwen2.5-Instruct-3B)

Given the capabilities of a 3B parameter model and the hardware advantage of **250GB NVRAM**, the MCP server will expose tools specifically designed for tasks where local, high-memory, continuous operation shines:

1. **Long-Running Process Monitoring**
   - **Scenario:** Watching a lengthy compilation (e.g., large C++ or Rust builds), long-running test suites, or training runs.
   - **Action:** The Qwen 3B agent continually ingests logs in real-time. With vast NVRAM, it can buffer and analyze massive outputs indefinitely without worrying about context window costs.
   - **Result:** It reports back to the main software factory only when the process crashes, succeeds, or hangs, providing a concise summary of the exact point of failure.

2. **Basic Health Checks & Log Triage**
   - **Scenario:** A dev server or background worker is running.
   - **Action:** Qwen 3B monitors the service's output stream, looking for stack traces, warnings, or error patterns.
   - **Result:** It filters out the noise and provides a distilled summary of anomalies. Crucially, when returning details to the primary reasoning model, it provides **file pointers** (e.g., exact file paths and line numbers) rather than dumping full file contents, allowing the larger model to selectively investigate code that lives remotely or in the software factory's environment.

3. **Dependency Checking & Environment Triage**
   - **Scenario:** A script fails due to a missing package, or the project environment needs verifying before a build.
   - **Action:** Qwen 3B intercepts "ModuleNotFoundError", "Cannot find module", or linker errors. It can inspect local lockfiles (`package.json`, `requirements.txt`) or run native check commands (like `pip check` or `npm install --dry-run`).
   - **Result:** It isolates the exact missing dependency or version mismatch. If it's a simple fix (like a missing import), it can flag it; for complex version resolution conflicts, it summarizes the conflict state and hands it back to the larger reasoning model to untangle.

## Architecture

1. **MCP Server Interface:** A lightweight Python service implementing the Model Context Protocol.
2. **Local Inference:** Qwen2.5-Instruct-3B running locally, handling the specific monitoring and summarization tasks requested via MCP.
3. **Software Factory Clients:** Cursor, Claude Code, or any other MCP-compatible client connects to this server.
4. **Workflow:** 
   - The primary AI (e.g., in Cursor) decides it needs to run a heavy task, like `make all`.
   - It calls an MCP tool like `monitor_long_process`.
   - The MCP server hands this off to the local Qwen 3B agent.
   - The primary AI can continue working on other tasks or simply wait, without wasting tokens on polling logs.
   - The Qwen 3B agent monitors the process, analyzes thousands of lines of output locally, and returns a highly distilled summary (e.g., "Build failed on line 405: missing header in auth.cpp") back through the MCP interface.