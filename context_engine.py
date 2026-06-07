import os
import gc
import chromadb
from sentence_transformers import SentenceTransformer
from config import NVME_DB_PATH

# Override HF Cache path to avoid permission issues
os.environ["HF_HOME"] = "/home/ameyades/agent_harness/hf_cache"

class ContextEngine:
    def __init__(self, db_path: str = NVME_DB_PATH):
        """
        Initializes a Context Manager for storing and retrieving codebase knowledge.
        HARDWARE CONSTRAINT: Explicitly initializes and persists its index files 
        directly to the NVMe path. Every write commits to disk to save Unified RAM.
        """
        os.makedirs(db_path, exist_ok=True)
        
        # PersistentClient writes immediately to the specified disk path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="agent_context")
        
        # Load small, low-overhead local embedding model
        print(f"[ContextEngine] Loading embedding model to Unified RAM...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"[ContextEngine] Embedding model loaded.")
        
    def _chunk_text_safely(self, text: str, max_tokens: int = 512) -> list:
        """
        Chunks incoming text safely into fixed windows.
        Uses a lightweight 4-chars-per-token heuristic to avoid heavy tokenizer dependencies.
        """
        chunk_size_chars = max_tokens * 4
        chunks = []
        for i in range(0, len(text), chunk_size_chars):
            chunks.append(text[i:i + chunk_size_chars])
        return chunks

    def add_to_context(self, text: str, metadata: dict = None):
        """
        Chunks text, embeds it using a local model, and commits it 
        to the NVMe vector database immediately.
        """
        if metadata is None:
            metadata = {}
            
        chunks = self._chunk_text_safely(text)
        if not chunks:
            return
            
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Generate deterministic IDs for chunks
        chunk_hash = hash(text)
        ids = [f"doc_{chunk_hash}_{i}" for i in range(len(chunks))]
        metadatas = [metadata for _ in chunks]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        # CRITICAL: Free up memory immediately after embedding (Unified RAM constraint)
        del chunks
        del embeddings
        del ids
        del metadatas
        gc.collect()

    def retrieve_context(self, query: str, top_k: int = 3) -> list:
        """
        Retrieves top_k context chunks from the NVMe database.
        """
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        documents = results.get("documents", [[]])[0]
        
        # Explicit Context Deletion
        del query_embedding
        del results
        gc.collect()
        
        return documents
