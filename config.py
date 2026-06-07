import os

# Hardware-constrained NVMe paths
NVME_BASE_PATH = os.environ.get("NVME_BASE_PATH", "/mnt/nvme")
NVME_DB_PATH = os.path.join(NVME_BASE_PATH, "context_db")
TELEMETRY_PATH = os.path.join(NVME_BASE_PATH, "router_telemetry.jsonl")

# Model configuration
LOCAL_MODEL_URL = "http://localhost:8080/v1"
LOCAL_MODEL_NAME = "qwen-3b"
STRONG_MODEL_NAME = "gpt-4o"

# Asyncio concurrency limit to prevent memory exhaustion
MAX_CONCURRENT_AGENTS = 10
