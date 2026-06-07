import os
import gc
from typing import Dict, Callable
from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 3) -> str:
    """
    Performs a lightweight web search using duckduckgo-search.
    Returns plain text strings instead of heavy Pydantic models.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "")
                body = r.get("body", "")
                results.append(f"Title: {title}\nSnippet: {body}\n---")
                
        compiled_results = "\n".join(results)
        
        # Explicit garbage collection
        del results
        gc.collect()
        
        return compiled_results if compiled_results else "No results found."
    except Exception as e:
        return f"Web search error: {str(e)}"

def read_local_logs(log_path: str, max_lines: int = 100) -> str:
    """
    Reads raw system strings from an NVMe path.
    """
    if not os.path.exists(log_path):
        return f"Error: Path {log_path} not found."
        
    try:
        # Read lines efficiently from the end
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        recent_lines = "".join(lines[-max_lines:])
        
        # Explicit memory cleanup
        del lines
        gc.collect()
        
        return recent_lines
    except Exception as e:
        return f"File read error: {str(e)}"


# Lightweight, plain-text tool execution schema
# Uses minimalist dictionary structures instead of heavy Pydantic validation schemas
TOOLCHAIN: Dict[str, Callable] = {
    "web_search": web_search,
    "read_local_logs": read_local_logs
}

def execute_tool(tool_name: str, **kwargs) -> str:
    """Minimalist tool dispatcher."""
    if tool_name not in TOOLCHAIN:
        return f"Error: Tool '{tool_name}' does not exist in the toolchain."
    
    return TOOLCHAIN[tool_name](**kwargs)
