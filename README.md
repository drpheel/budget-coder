# Async Agent Harness

A production-grade, asynchronous Agent Harness built for hardware-constrained environments like the NVIDIA Jetson Orin Nano (8GB Unified RAM).

## Features
- **Asyncio Swarm Foundation**: Lightweight `Agent` class scaling up to 10 concurrent agents via `asyncio.Semaphore`.
- **Aggressive Garbage Collection**: Explicit memory teardown (`del` and `gc.collect()`) after execution loops to preserve unified RAM.
- **RouteLLM Traffic Controller**: Implements Matrix Factorization (`mf`) routing with an absolute network fallback block. Re-routes cloud model timeouts natively back to local LLM deployments.
- **NVMe-Persisted RAG**: `ChromaDB` configured strictly for physical disk IO instead of in-memory. Employs `sentence-transformers` lightweight model.
- **System Telemetry**: Records every prompt payload, choice, latency, and success status to NVMe JSONL streams for future router fine-tuning.
- **Lightweight Toolchain**: Pure dictionary-based execution mapping. Native web search and system file readers.

## Running the Harness
```bash
pip install -r requirements.txt
python main.py
```