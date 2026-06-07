import asyncio
import gc
import os
from config import NVME_DB_PATH, TELEMETRY_PATH, NVME_BASE_PATH
from telemetry import TelemetryLogger
from context_engine import ContextEngine
from router import TrafficController
from agent import Agent
from tools import execute_tool

async def main():
    """
    The Main Execution Loop.
    Initializes components, receives prompts, gathers context via RAG & Tools,
    routes to LLMs, and streams output.
    """
    print("="*60)
    print(" ASYNC AGENT HARNESS (NVIDIA Jetson Orin Nano, 8GB Unified) ")
    print("="*60)
    
    # Ensure physical disk paths are ready
    os.makedirs(NVME_BASE_PATH, exist_ok=True)
    
    # 1. Initialize Telemetry Logger (NVMe persistence)
    print("[Main] Initializing Telemetry Logger on NVMe...")
    telemetry = TelemetryLogger(TELEMETRY_PATH)
    
    # 2. Initialize RouteLLM Traffic Controller with Fallbacks
    print("[Main] Initializing RouteLLM Traffic Controller...")
    router = TrafficController(telemetry)
    
    # 3. Initialize Context Engine (NVMe Persisted RAG)
    print("[Main] Initializing Context Engine...")
    context_engine = ContextEngine(NVME_DB_PATH)
    
    # 4. Initialize the Asyncio Swarm Foundation
    print("[Main] Initializing Asyncio Agent Pool...")
    agent = Agent(router)
    
    # Seed some dummy context immediately
    print("[Main] Seeding initial architecture knowledge to NVMe DB...")
    context_engine.add_to_context(
        "NVIDIA Jetson Orin Nano features 8GB of Unified RAM shared between CPU and GPU. Aggressive garbage collection is required.",
        metadata={"source": "hardware_specs"}
    )
    
    # Define execution payloads (environment prompts)
    tasks = [
        {"id": "TASK_1", "query": "What are the core hardware memory constraints of my system?"},
        {"id": "TASK_2", "query": "Check the local router telemetry logs and summarize the last few lines."}
    ]
    
    print("\n[Main] Starting Execution Loop...")
    
    for task in tasks:
        task_id = task["id"]
        query = task["query"]
        
        print(f"\n--- Processing {task_id} ---")
        print(f"Query: {query}")
        
        # 5. Tool Context Retrieval (Lightweight Dictionary Schema)
        dynamic_context = []
        
        if "logs" in query.lower() or "telemetry" in query.lower():
            print(f"[{task_id}] Triggering read_local_logs tool...")
            log_data = execute_tool("read_local_logs", log_path=TELEMETRY_PATH, max_lines=5)
            dynamic_context.append(f"[Local Telemetry Logs]\n{log_data}")
            
        if "news" in query.lower():
            print(f"[{task_id}] Triggering web_search tool...")
            search_data = execute_tool("web_search", query="AI Edge computing news", max_results=2)
            dynamic_context.append(f"[Web Search Results]\n{search_data}")
            
        # 6. Retrieve RAG Context from NVMe
        print(f"[{task_id}] Retrieving Context from NVMe Database...")
        rag_data = context_engine.retrieve_context(query, top_k=2)
        if rag_data:
            dynamic_context.extend(rag_data)
            
        # 7. Execute Agent payload
        print(f"[{task_id}] Dispatching unified payload to Agent Swarm...")
        result = await agent.process_task(
            task_id=task_id,
            prompt=query,
            context=dynamic_context
        )
        
        # Stream the final output
        print(f"\n[{task_id} FINAL OUTPUT STREAM]")
        print("-" * 40)
        print(result)
        print("-" * 40)
        
        # 8. Main Loop Explicit Deletion
        del dynamic_context
        del rag_data
        gc.collect()

    print("\n[Main] Execution loop fully completed. System idling.")

if __name__ == "__main__":
    asyncio.run(main())
