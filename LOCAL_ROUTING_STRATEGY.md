# Local Routing Strategy: Replacing OpenAI with Qwen

Currently, the Agent Harness uses **RouteLLM** with the `mf` (Matrix Factorization) router. While the `mf` router computes the routing decision *locally* (without an API call), it ultimately routes the prompt to either the local weak model (`qwen-3b`) or the external cloud strong model (`gpt-4o` via OpenAI). 

If you want to completely sever the dependency on OpenAI and use Qwen models for *everything* (both the routing decisions and the actual generation), here is how you can architect it.

## Approach 1: Dual-Local Models via LiteLLM (The RouteLLM Way)

Since RouteLLM uses LiteLLM under the hood, you can map both the "weak" and "strong" models to local OpenAI-compatible endpoints (like `llama-server`). 

If you have a second device (or a larger server) running a heavier model like **Qwen 14B or 72B**, you can set that up as the strong model.

**1. Update `config.py`:**
```python
LOCAL_WEAK_URL = "http://localhost:8080/v1"
LOCAL_STRONG_URL = "http://<LOCAL_NETWORK_IP>:8081/v1"

WEAK_MODEL_NAME = "qwen-3b"
STRONG_MODEL_NAME = "qwen-14b"
```

**2. Update `router.py` to map both endpoints:**
```python
import litellm
import os

# Tell LiteLLM where to find both models
litellm.api_base = LOCAL_WEAK_URL
os.environ["OPENAI_API_BASE"] = LOCAL_WEAK_URL # Default fallback

# Define Custom Endpoints in the Controller
self.controller = Controller(
    routers=["mf"],
    strong_model=f"openai/{STRONG_MODEL_NAME}",
    weak_model=f"openai/{WEAK_MODEL_NAME}",
    # LiteLLM allows passing base_urls dynamically for different models
    litellm_kwargs={
        f"openai/{STRONG_MODEL_NAME}": {"api_base": LOCAL_STRONG_URL},
        f"openai/{WEAK_MODEL_NAME}": {"api_base": LOCAL_WEAK_URL}
    }
)
```

---

## Approach 2: LLM-as-a-Judge (Pure Qwen Routing)

If you don't want to use the `routellm` package at all, you can use the **Qwen 3B model itself to act as the traffic controller**. 

Because Qwen 3B is incredibly fast, you can add a "Triage Step" where the agent asks Qwen to evaluate the complexity of the prompt before actually answering it. 

### How it works:
1. The user submits a prompt.
2. The Agent sends a fast, restricted-token request to Qwen 3B asking it to output a single JSON classification: `{"complexity": "low", "requires_web": true}`.
3. If complexity is low, Qwen 3B answers it. If complexity is high, the system defers the prompt to a remote Qwen instance or queues it for a longer thought-process (like triggering an extended-reasoning loop).

### Implementation (`router.py` replacement):

```python
import aiohttp
import json

class QwenRouter:
    def __init__(self, local_url="http://localhost:8080/v1"):
        self.local_url = local_url
        
    async def _evaluate_complexity(self, prompt: str, session: aiohttp.ClientSession) -> bool:
        """
        Uses Qwen 3B to decide if the prompt is hard.
        Returns True if complex, False if simple.
        """
        triage_prompt = f"""
        Evaluate the complexity of this task. If it requires deep reasoning, complex code generation, or heavy logic, respond with exactly {"complex": true}. Otherwise, {"complex": false}.
        Task: {prompt}
        """
        payload = {
            "model": "qwen-3b",
            "messages": [{"role": "user", "content": triage_prompt}],
            "temperature": 0.1,
            "max_tokens": 10 # We only need a tiny JSON response
        }
        
        async with session.post(f"{self.local_url}/chat/completions", json=payload) as response:
            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            return "true" in content.lower()

    async def generate(self, prompt: str) -> str:
        async with aiohttp.ClientSession() as session:
            # 1. Ask Qwen to route the prompt
            is_complex = await self._evaluate_complexity(prompt, session)
            
            # 2. Route execution based on Qwen's own decision
            if is_complex:
                print("[Router] Qwen deemed this task COMPLEX. Routing to Heavy API...")
                return await self._call_heavy_model(prompt, session)
            else:
                print("[Router] Qwen deemed this task SIMPLE. Handling locally...")
                return await self._call_local_model(prompt, session)
```

### Why Approach 2 is great for the Jetson:
- **No `routellm` dependency**: Saves RAM by removing the Matrix Factorization libraries and LightGBM models.
- **Dynamic Capabilities**: You can ask Qwen to not only route by complexity, but to dynamically decide *which tools* to use in the same triage pass. 
- **100% Local**: No API keys, no network timeouts, no dependency on external AI platforms.