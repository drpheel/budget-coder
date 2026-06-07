import asyncio
import aiohttp
import gc
from router import TrafficController
from config import MAX_CONCURRENT_AGENTS

class Agent:
    def __init__(self, traffic_controller: TrafficController):
        """
        Lightweight async Agent using asyncio and aiohttp.
        """
        self.traffic_controller = traffic_controller
        # Semaphore implementation to scale concurrent agents without blocking the execution thread
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_AGENTS)
        
    async def process_task(self, task_id: str, prompt: str, context: list) -> str:
        """
        Processes a single task through the routing pipeline.
        Enforces strict memory management for constrained hardware environments.
        """
        async with self.semaphore:
            print(f"[Agent Task {task_id}] Acquiring execution slot...")
            
            # Compile the contextual payload
            context_string = "\n".join(context) if context else "No additional context."
            full_prompt = f"System Context:\n{context_string}\n\nUser Task:\n{prompt}"
            
            response_text = ""
            # Use aiohttp ClientSession for the underlying request logic
            async with aiohttp.ClientSession() as session:
                response_text = await self.traffic_controller.generate_with_routing(
                    prompt=full_prompt, 
                    session=session
                )
            
            print(f"[Agent Task {task_id}] Execution complete.")
            
            # ------------------------------------------------------------------
            # CRITICAL: Aggressive manual garbage collection and explicit 
            # context deletion to prevent Unified RAM accumulation over time.
            # ------------------------------------------------------------------
            del context_string
            del full_prompt
            del context
            del session
            
            # Force the garbage collector to immediately release deleted objects
            gc.collect()
            
            return response_text
