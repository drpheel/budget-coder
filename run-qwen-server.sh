#!/usr/bin/env bash
set -euo pipefail
LLAMA_DIR="$HOME/llama.cpp"
MODEL="$HOME/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
BIN="$LLAMA_DIR/build/bin/llama-server"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$LLAMA_DIR/build/bin:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$BIN" ]]; then
  echo "llama-server not built yet."
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL"
  exit 1
fi

if systemctl is-active --quiet ollama 2>/dev/null; then
  echo "ERROR: ollama is running and will cause GPU OOM."
  echo "Run: sudo systemctl stop ollama"
  exit 1
fi

if pgrep -u "$USER" -f 'llama-server' >/dev/null 2>&1; then
  echo "Stopping stale llama-server processes..."
  pkill -u "$USER" -f 'llama-server' || true
  sleep 1
fi

echo "Loading 32k context with full GPU offload (may OOM on 8GB Jetson)..."
echo "Starting OpenAI-compatible HTTP server on http://localhost:8080/v1"
# -fit on will reduce layers/context if 32k + ngl 99 does not fit
exec "$BIN" -m "$MODEL" -fit on -ngl 99 -c 32768 -np 1 \
  -ctk q4_0 -ctv q4_0 \
  -b 256 \
  --host 127.0.0.1 --port 8080 \
  "$@"
