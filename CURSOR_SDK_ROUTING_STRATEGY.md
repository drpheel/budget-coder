# Cursor SDK Routing Strategy: Maximizing the $20 Subscription

The ultimate goal of this embedded Agent Harness is to provide a massive multiplier to a standard $20/month Cursor subscription. By handling 90% of simple tasks (log reading, web searches, basic questions) completely locally on the Jetson's Unified RAM using `qwen-3b`, we can save your precious cloud compute limits. 

For the 10% of tasks that require heavy code refactoring, complex reasoning, or multi-file awareness, the local router can seamlessly offload the payload to the **Cursor SDK**. This turns your Jetson into a dedicated triage manager that only wakes up the "expensive" cloud brain when absolutely necessary.

Here is exactly how you can implement this hybrid local-to-SDK routing architecture.

## 1. Install the Cursor SDK

First, add the official Cursor SDK to your Python environment:

```bash
cd ~/agent_harness
source venv/bin/activate
pip install cursor-sdk
```

## 2. Update `router.py` to use `cursor_sdk` for Complex Tasks

We will replace the `RouteLLM` dependency (which routes between two basic OpenAI-compatible endpoints) with our own **Qwen Triage Router**. This router will ask the local Qwen model to rate the complexity of the prompt. If it's too complex, we trigger `Agent.prompt()` from the Cursor SDK.

```python
import aiohttp
import asyncio
import os
from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

class TrafficController:
    def __init__(self, telemetry_logger):
        self.telemetry = telemetry_logger
        self.local_url = "http://localhost:8080/v1"
        
        # Ensure the Cursor API key is available
        self.cursor_api_key = os.environ.get("CURSOR_API_KEY")

    async def _evaluate_complexity(self, prompt: str, session: aiohttp.ClientSession) -> bool:
        """
        Uses the local Qwen 3B model to decide if the prompt is complex.
        Returns True if complex (needs Cursor SDK), False if simple (handled locally).
        """
        triage_prompt = f"""
        Evaluate the complexity of this software engineering task. 
        If it requires deep reasoning, writing complex code, multi-file refactoring, or heavy logic, respond with EXACTLY the word "COMPLEX". 
        If it is a simple question, log reading, or basic web search, respond with EXACTLY the word "SIMPLE".
        
        Task: {prompt}
        """
        
        payload = {
            "model": "qwen-3b",
            "messages": [{"role": "user", "content": triage_prompt}],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        try:
            async with session.post(f"{self.local_url}/chat/completions", json=payload) as response:
                data = await response.json()
                content = data["choices"][0]["message"]["content"].strip().upper()
                return "COMPLEX" in content
        except Exception as e:
            print(f"[Triage Error] Failed to evaluate complexity: {e}. Defaulting to COMPLEX.")
            return True

    async def _call_cursor_sdk(self, prompt: str) -> str:
        """
        Offloads the heavy lifting to the Cursor SDK.
        This uses Cursor's cloud intelligence to solve the problem directly.
        """
        if not self.cursor_api_key:
            return "Error: CURSOR_API_KEY environment variable is missing. Cannot route to SDK."
            
        print("[Router] Dispatching to Cursor SDK (Cloud Intelligence)...")
        
        # We wrap the synchronous Cursor SDK call in an async executor to prevent blocking
        loop = asyncio.get_event_loop()
        
        def run_cursor_agent():
            # One-shot execution against the local workspace
            result = Agent.prompt(
                prompt,
                AgentOptions(
                    api_key=self.cursor_api_key,
                    model="composer-2.5", # Let Cursor use its frontier models
                    local=LocalAgentOptions(cwd=os.getcwd()),
                )
            )
            return result

        try:
            result = await loop.run_in_executor(None, run_cursor_agent)
            
            if result.status == "error":
                return f"Cursor SDK Run Failed (ID: {result.id})"
                
            return f"[Cursor SDK Answer]\n{result.result}"
            
        except Exception as err:
            return f"Cursor SDK Startup Failed: {str(err)}"

    async def _call_local_model(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """
        Handles the prompt entirely locally using the Jetson's Qwen 3B.
        """
        payload = {
            "model": "qwen-3b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        
        async with session.post(f"{self.local_url}/chat/completions", json=payload) as response:
            data = await response.json()
            return data["choices"][0]["message"]["content"]

    async def generate_with_routing(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """
        The main routing entrypoint.
        """
        import time
        start_time = time.time()
        
        # 1. Triage the complexity
        is_complex = await self._evaluate_complexity(prompt, session)
        
        # 2. Route the execution
        if is_complex:
            model_chosen = "cursor-sdk"
            response_text = await self._call_cursor_sdk(prompt)
        else:
            model_chosen = "local-qwen-3b"
            response_text = await self._call_local_model(prompt, session)
            
        latency = time.time() - start_time
        
        # 3. Log the decision
        asyncio.create_task(
            self.telemetry.log_routing_decision(
                prompt=prompt,
                model_chosen=model_chosen,
                latency_seconds=latency,
                execution_success=True
            )
        )
        
        return response_text
```

## Why this Architecture is Perfect for your Hardware

1. **Massive Cost Savings**: You never ping the Cursor SDK API for simple terminal commands, reading telemetry files, or asking basic context questions.
2. **Unified RAM Preservation**: The heavy `routellm` and `scikit-learn` dependencies are entirely removed. The only thing running in RAM is the native `llama-server` and a lightweight Python script.
3. **Cursor Context Awareness**: Because the `Agent.prompt()` SDK call is configured with `local=LocalAgentOptions(cwd=os.getcwd())`, the Cursor cloud agent has full read/write capabilities over your local Jetson filesystem. If it needs to write a new script or refactor a file, it will do it natively on your device.
4. **Resiliency**: If the Jetson completely loses internet access, the router will fail to hit the Cursor SDK, but you could easily wrap the `_call_cursor_sdk` function in a `try/except` block to automatically fallback to `_call_local_model` for 100% offline uptime.