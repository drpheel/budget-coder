import time
import asyncio
import aiohttp
import os
from routellm.controller import Controller
from config import LOCAL_MODEL_URL, LOCAL_MODEL_NAME, STRONG_MODEL_NAME
from telemetry import TelemetryLogger

class TrafficController:
    def __init__(self, telemetry_logger: TelemetryLogger):
        self.telemetry = telemetry_logger
        
        # Define weak and strong models. RouteLLM leverages LiteLLM.
        self.weak_model_name = f"openai/{LOCAL_MODEL_NAME}"
        self.strong_model_name = STRONG_MODEL_NAME
        
        # Configure LiteLLM (under RouteLLM) to route the weak model to our local Qwen 3B API
        os.environ["OPENAI_API_BASE"] = LOCAL_MODEL_URL
        
        # Ensure a dummy key exists to bypass basic LiteLLM client validation if missing
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
            
        try:
            # Integrate routellm package using the Matrix Factorization (mf) router
            self.controller = Controller(
                routers=["mf"],
                strong_model=self.strong_model_name,
                weak_model=self.weak_model_name
            )
        except Exception as e:
            print(f"[Router Initialization Warning] {e}. Using pure fallback routing.")
            self.controller = None

    async def _call_local_model_fallback(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """
        Directly calls the local Qwen 3B model via OpenAI-compatible endpoint.
        Used when the primary router or cloud endpoint fails.
        """
        payload = {
            "model": LOCAL_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-local-dummy"
        }
        
        async with session.post(f"{LOCAL_MODEL_URL}/chat/completions", json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["message"]["content"]

    async def generate_with_routing(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """
        Executes routing via RouteLLM. Includes a critical network fallback.
        """
        start_time = time.time()
        success = False
        model_chosen = "unknown"
        response_text = ""
        
        try:
            if not self.controller:
                raise Exception("RouteLLM Controller not initialized")
                
            # Run the synchronous RouteLLM request in a thread executor to prevent blocking the async loop
            loop = asyncio.get_event_loop()
            
            def sync_routellm_call():
                # 'router-mf-0.11593' represents the MF router in RouteLLM
                return self.controller.chat.completions.create(
                    model="router-mf-0.11593",
                    messages=[{"role": "user", "content": prompt}]
                )
                
            response = await loop.run_in_executor(None, sync_routellm_call)
            
            # The model attribute indicates which model RouteLLM selected
            model_chosen = getattr(response, 'model', 'routed_model')
            response_text = response.choices[0].message.content
            success = True
            
        except Exception as e:
            # CRITICAL NETWORK FALLBACK: 
            # If cloud API request fails due to internet timeout, disconnect, or credential issue,
            # catch the exception, log a warning, and force-route to the local Qwen 3B model.
            print(f"[Router Fallback Warning] Cloud/Router request failed: {e}. Forcing local {LOCAL_MODEL_NAME} execution.")
            model_chosen = f"fallback_{LOCAL_MODEL_NAME}"
            
            try:
                response_text = await self._call_local_model_fallback(prompt, session)
                success = True
            except Exception as local_e:
                print(f"[Router FATAL] Local model fallback also failed: {local_e}")
                response_text = f"System Error: Unable to reach any model. Fallback error: {local_e}"
                success = False
                
        latency = time.time() - start_time
        
        # Log telemetry asynchronously to ensure we don't block the returning payload
        asyncio.create_task(
            self.telemetry.log_routing_decision(
                prompt=prompt,
                model_chosen=model_chosen,
                latency_seconds=latency,
                execution_success=success
            )
        )
        
        return response_text
