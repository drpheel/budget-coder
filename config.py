import os

# Set Hugging Face cache absolute path at the very top
os.environ["HF_HOME"] = "/home/ameyades/agent_harness/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/ameyades/agent_harness/hf_cache"

# Hardware-constrained NVMe paths
# Defaulting to local directories since /mnt/nvme requires root/sudo access
NVME_BASE_PATH = os.environ.get("NVME_BASE_PATH", "/home/ameyades/agent_harness/nvme_mock")
NVME_DB_PATH = os.path.join(NVME_BASE_PATH, "context_db")
TELEMETRY_PATH = os.path.join(NVME_BASE_PATH, "router_telemetry.jsonl")

# Model configuration
LOCAL_MODEL_URL = "http://localhost:8080/v1"
LOCAL_MODEL_NAME = "qwen-3b"
STRONG_MODEL_NAME = "gpt-4o"

# Asyncio concurrency limit to prevent memory exhaustion
MAX_CONCURRENT_AGENTS = 10
