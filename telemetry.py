import json
import time
import os
import asyncio
from config import TELEMETRY_PATH

class TelemetryLogger:
    def __init__(self, log_path: str = TELEMETRY_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # Prevent race conditions in asynchronous file writing
        self.lock = asyncio.Lock()
        
    async def log_routing_decision(self, prompt: str, model_chosen: str, latency_seconds: float, execution_success: bool):
        """
        Appends a structured JSON row to the NVMe file for EVERY routing decision.
        """
        # Truncate and sanitize prompt for logging
        prompt_preview = prompt[:100].replace('\n', ' ') + "..." if len(prompt) > 100 else prompt.replace('\n', ' ')
        
        log_entry = {
            "timestamp": time.time(),
            "prompt_preview": prompt_preview,
            "model_chosen": model_chosen,
            "latency_seconds": round(latency_seconds, 4),
            "execution_success_bool": execution_success
        }
        
        # Async lock to safely append to the telemetry file
        async with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                f.flush()
                # Ensure data is immediately synced to the NVMe drive
                os.fsync(f.fileno())
